"""Per-shape transcript assignment and table construction."""

from __future__ import annotations

import logging
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import get_transformation
from tqdm.auto import tqdm

from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
    write_spatialdata_zarr,
)
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    ORIGINAL_ID_NAMESPACE,
    PROSEG_ASSIGNMENT_COLUMN,
    PROSEG_ID_NAMESPACE,
    SOURCE_CELL_ID_COLUMN,
    canonical_instance_series,
    choose_primary_points_key,
    register_segmentation_branch,
    stamp_merxen_schema,
)
from merxen.io.transcript_io import first_existing_col, iter_points_chunks
from merxen.memory import enforce_memory_limit, force_release, log_status

logger = logging.getLogger(__name__)


def sanitize_table_key(shape_key: str, table_prefix: str = "table_") -> str:
    """Build a safe table key from a shape key."""
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(shape_key)).strip("_")
    return f"{table_prefix}{safe}"


def resolve_points_cols(points_obj: Any) -> tuple[str, str, str, str | None]:
    """Resolve x/y/gene/qv columns from a points table."""
    x_col = first_existing_col(
        points_obj,
        ["x", "x_micron", "x_location", "global_x", "x_global_px", "observed_x"],
    )
    y_col = first_existing_col(
        points_obj,
        ["y", "y_micron", "y_location", "global_y", "y_global_px", "observed_y"],
    )
    gene_col = first_existing_col(points_obj, ["gene", "feature_name", "target"])
    qv_col = first_existing_col(
        points_obj,
        ["transcript_score", "qv", "quality", "quality_value"],
    )

    if x_col is None or y_col is None or gene_col is None:
        raise KeyError(
            "Could not resolve points columns. "
            f"x={x_col}, y={y_col}, gene={gene_col}. "
            f"Available: {list(points_obj.columns)}"
        )
    return x_col, y_col, gene_col, qv_col


def ensure_shape_has_cell_id(
    sdata_obj: Any,
    shape_key: str,
) -> tuple[gpd.GeoDataFrame, str]:
    """Return a shape GeoDataFrame with canonical positive instance IDs."""
    shp = sdata_obj.shapes[shape_key]
    if INSTANCE_ID_COLUMN in shp.columns:
        gdf = shp.copy()
        instance_ids = canonical_instance_series(
            gdf[INSTANCE_ID_COLUMN],
            field_name=f"{shape_key}.{INSTANCE_ID_COLUMN}",
        )
        if bool(instance_ids.duplicated().any()):
            raise ValueError(f"[{shape_key}] Duplicate instance IDs")
        gdf[INSTANCE_ID_COLUMN] = instance_ids.to_numpy(dtype=np.uint64)
        gdf.index = pd.Index(
            gdf[INSTANCE_ID_COLUMN],
            dtype="uint64",
            name=INSTANCE_ID_COLUMN,
        )
        return gdf, INSTANCE_ID_COLUMN

    candidate = first_existing_col(
        shp,
        [
            SOURCE_CELL_ID_COLUMN,
            "cell_id",
            "cell",
            "cells",
            "cell_ID",
            "region",
            "label_id",
        ],
    )

    gdf = shp.copy()
    source_ids = (
        gdf[candidate].astype(str)
        if candidate is not None
        else pd.Series(gdf.index.astype(str), index=gdf.index)
    )
    if bool(source_ids.duplicated().any()):
        raise ValueError(
            f"[{shape_key}] Duplicate source cell IDs cannot be normalized safely"
        )
    numeric = pd.to_numeric(source_ids, errors="coerce")
    if bool(numeric.notna().all()) and bool((numeric >= 0).all()):
        if bool((numeric == 0).any()):
            numeric = numeric + 1
        instance_ids = numeric.astype("uint64")
    else:
        ordered = {
            source_id: index + 1
            for index, source_id in enumerate(sorted(source_ids.tolist()))
        }
        instance_ids = source_ids.map(ordered).astype("uint64")
    gdf[SOURCE_CELL_ID_COLUMN] = source_ids.to_numpy(dtype=object)
    gdf[INSTANCE_ID_COLUMN] = instance_ids.to_numpy(dtype=np.uint64)
    gdf.index = pd.Index(
        gdf[INSTANCE_ID_COLUMN],
        dtype="uint64",
        name=INSTANCE_ID_COLUMN,
    )
    return gdf, INSTANCE_ID_COLUMN


def build_gene_list_from_base_table(sdata_obj: Any) -> list[str]:
    """Extract stable gene vocabulary from ``sdata.tables['table']``."""
    if "table" not in sdata_obj.tables:
        raise KeyError("Expected sdata.tables['table'] to exist for gene vocabulary.")

    base_tbl = sdata_obj.tables["table"]
    if "gene" in base_tbl.var.columns:
        genes = base_tbl.var["gene"].astype(str)
    else:
        genes = base_tbl.var_names.astype(str)

    gene_index = pd.Index(pd.Series(genes).dropna().astype(str).unique())
    return [str(g) for g in gene_index.sort_values().tolist()]


def clone_table_for_region(table_obj: ad.AnnData, region_name: str) -> ad.AnnData:
    """Clone an existing table and retarget it to a different shape region."""
    table_copy = table_obj.copy()
    attrs = dict(table_copy.uns.get("spatialdata_attrs", {}))
    region_key = str(attrs.get("region_key", "region"))
    instance_key = attrs.get("instance_key")

    if region_key not in table_copy.obs.columns:
        region_key = "region"
        table_copy.obs[region_key] = region_name
    table_copy.obs[region_key] = pd.Categorical([region_name] * table_copy.n_obs)

    if (instance_key is None) or (instance_key not in table_copy.obs.columns):
        for cand in ["cell_id", "cell", "cells", "cell_ID"]:
            if cand in table_copy.obs.columns:
                instance_key = cand
                break
    if (instance_key is None) or (instance_key not in table_copy.obs.columns):
        instance_key = SOURCE_CELL_ID_COLUMN
        table_copy.obs[instance_key] = table_copy.obs_names.astype(str)

    source_ids = table_copy.obs[instance_key].astype(str)
    if bool(source_ids.duplicated().any()):
        raise ValueError("Table contains duplicate source instance identifiers")
    if instance_key == INSTANCE_ID_COLUMN:
        instance_ids = canonical_instance_series(
            table_copy.obs[INSTANCE_ID_COLUMN],
            field_name=f"{region_name}.{INSTANCE_ID_COLUMN}",
        )
    else:
        numeric = pd.to_numeric(source_ids, errors="coerce")
        if bool(numeric.notna().all()) and bool((numeric >= 0).all()):
            if bool((numeric == 0).any()):
                numeric = numeric + 1
            instance_ids = numeric.astype("uint64")
        else:
            ordered = {
                source_id: index + 1
                for index, source_id in enumerate(sorted(source_ids.tolist()))
            }
            instance_ids = source_ids.map(ordered).astype("uint64")
        table_copy.obs[SOURCE_CELL_ID_COLUMN] = source_ids.to_numpy(dtype=object)
    table_copy.obs[INSTANCE_ID_COLUMN] = instance_ids.to_numpy(dtype=np.uint64)
    table_copy.obs_names = table_copy.obs[INSTANCE_ID_COLUMN].astype(str).to_numpy()
    table_copy.uns.pop("spatialdata_attrs", None)
    return TableModel.parse(
        table_copy,
        region=region_name,
        region_key=region_key,
        instance_key=INSTANCE_ID_COLUMN,
    )


def compute_table_from_points_for_shape(
    dataset_name: str,
    points_obj: Any,
    shape_gdf: gpd.GeoDataFrame,
    shape_id_col: str,
    shape_key: str,
    gene_list: list[str],
    *,
    chunk_rows: int = 750_000,
    status_every_chunks: int = 5,
    memory_check_every_chunks: int = 5,
    max_ram_gb: float = 600.0,
    warn_ram_gb: float = 560.0,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Assign points to one shape layer and build an AnnData count table."""
    x_col, y_col, gene_col, _ = resolve_points_cols(points_obj)

    gdf_shapes = shape_gdf[[shape_id_col, "geometry"]].copy()
    gdf_shapes = gdf_shapes[
        gdf_shapes.geometry.notna() & ~gdf_shapes.geometry.is_empty
    ].copy()
    gdf_shapes[shape_id_col] = canonical_instance_series(
        gdf_shapes[shape_id_col],
        field_name=f"{shape_key}.{shape_id_col}",
    ).to_numpy(dtype=np.uint64)

    cell_ids = gdf_shapes[shape_id_col].astype("uint64").tolist()
    cell_to_idx = {int(cell_id): i for i, cell_id in enumerate(cell_ids)}
    genes = [str(g) for g in gene_list]
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    _ = gdf_shapes.sindex
    counts_csr = sparse.csr_matrix((len(cell_ids), len(genes)), dtype=np.int64)

    n_input = 0
    n_used = 0
    n_assigned = 0

    chunk_iter = iter_points_chunks(
        points_obj,
        columns=[x_col, y_col, gene_col],
        chunk_rows=chunk_rows,
        desc=f"[{dataset_name}:{shape_key}] assign chunks",
    )
    for i, chunk in enumerate(chunk_iter, start=1):
        n_input += len(chunk)

        xv = pd.to_numeric(chunk[x_col], errors="coerce").to_numpy(np.float64)
        yv = pd.to_numeric(chunk[y_col], errors="coerce").to_numpy(np.float64)
        gv = chunk[gene_col].astype(str).to_numpy(dtype=object)
        valid = np.isfinite(xv) & np.isfinite(yv)
        valid &= pd.notna(gv)
        valid &= gv != ""

        if np.any(valid):
            x_valid = xv[valid]
            y_valid = yv[valid]
            g_valid = gv[valid]

            points = gpd.GeoDataFrame(
                {"gene": pd.Series(g_valid, dtype=str)},
                geometry=gpd.points_from_xy(x_valid, y_valid),
                crs=gdf_shapes.crs,
            ).reset_index(drop=True)

            shapes_subset = gdf_shapes[[shape_id_col, "geometry"]].reset_index(
                drop=True
            )

            joined = gpd.sjoin(
                points,
                shapes_subset,
                how="left",
                predicate="within",
            )

            cell_series = pd.to_numeric(
                joined[shape_id_col],
                errors="coerce",
            ).astype("UInt64")
            assigned_mask = cell_series.notna()
            if assigned_mask.any():
                assigned_cells = cell_series.loc[assigned_mask].to_numpy(
                    dtype=np.uint64,
                )
                assigned_genes = (
                    joined.loc[assigned_mask, "gene"].astype(str).to_numpy(dtype=object)
                )

                cidx = np.fromiter(
                    (cell_to_idx.get(int(cell), -1) for cell in assigned_cells),
                    dtype=np.int64,
                    count=len(assigned_cells),
                )
                gidx = np.fromiter(
                    (gene_to_idx.get(gene, -1) for gene in assigned_genes),
                    dtype=np.int64,
                    count=len(assigned_genes),
                )
                keep = (cidx >= 0) & (gidx >= 0)
                if np.any(keep):
                    data = np.ones(int(np.sum(keep)), dtype=np.int64)
                    chunk_mat = sparse.coo_matrix(
                        (data, (cidx[keep], gidx[keep])),
                        shape=(len(cell_ids), len(genes)),
                    ).tocsr()
                    counts_csr = counts_csr + chunk_mat
                    n_assigned += int(np.sum(keep))

            n_used += len(x_valid)
            del points, joined, x_valid, y_valid, g_valid

        if i % int(status_every_chunks) == 0:
            pct = 100.0 * n_assigned / max(n_used, 1)
            log_status(
                f"[{dataset_name}:{shape_key}] chunk={i} "
                f"input={n_input:,} used={n_used:,} "
                f"assigned={n_assigned:,} ({pct:.2f}%)"
            )

        if i % int(memory_check_every_chunks) == 0:
            enforce_memory_limit(
                stage=f"{dataset_name}:{shape_key} chunk {i}",
                max_gb=max_ram_gb,
                warn_gb=warn_ram_gb,
            )
            force_release()

        del chunk, xv, yv, gv, valid

    obs = pd.DataFrame(
        index=pd.Index(
            np.asarray(cell_ids, dtype=np.uint64).astype(str),
            dtype=str,
            name="obs_id",
        )
    )
    obs[INSTANCE_ID_COLUMN] = np.asarray(cell_ids, dtype=np.uint64)
    obs["region"] = pd.Categorical([shape_key] * len(obs), categories=[shape_key])

    var = pd.DataFrame(index=pd.Index(genes, dtype=str, name="gene"))
    var["gene"] = var.index.astype(str)

    adata = ad.AnnData(X=counts_csr, obs=obs, var=var)
    table = TableModel.parse(
        adata,
        region=shape_key,
        region_key="region",
        instance_key=INSTANCE_ID_COLUMN,
    )

    summary = {
        "dataset": dataset_name,
        "shape_key": shape_key,
        "n_cells": int(len(cell_ids)),
        "n_genes": int(len(genes)),
        "n_points_input": int(n_input),
        "n_points_used": int(n_used),
        "n_points_assigned": int(n_assigned),
        "pct_assigned": float(100.0 * n_assigned / max(n_used, 1)),
    }
    return table, summary


def shape_to_existing_table_source(shape_key: str) -> str | None:
    """Map shape names to reusable existing table sources when possible."""
    if shape_key in {"MOSAIK_proseg", "cell_boundaries"}:
        return "table"
    if shape_key in {"merscope_cell_boundaries", "xenium_cell_boundaries"}:
        return "table_original"
    return None


def _register_assignment_branch(
    sdata_obj: Any,
    *,
    points_key: str,
    shape_key: str,
    table_key: str,
) -> None:
    """Register a completed per-shape table using explicit branch semantics."""
    aliases: tuple[str, ...]
    if shape_key == "MOSAIK_proseg":
        branch = "proseg"
        assignment_column = (
            PROSEG_ASSIGNMENT_COLUMN
            if PROSEG_ASSIGNMENT_COLUMN in sdata_obj.points[points_key].columns
            else None
        )
        background_column = (
            "background"
            if "background" in sdata_obj.points[points_key].columns
            else None
        )
        namespace = PROSEG_ID_NAMESPACE
        aliases = ("reseg",)
    elif shape_key == "MOSAIK_cellpose":
        branch = "cellpose"
        assignment_column = None
        background_column = None
        namespace = PROSEG_ID_NAMESPACE
        aliases = ("proseg_mask", "cellpose_mask")
    elif shape_key in {"merscope_cell_boundaries", "xenium_cell_boundaries"}:
        branch = "original"
        assignment_column = (
            "original_assignment"
            if "original_assignment" in sdata_obj.points[points_key].columns
            else None
        )
        background_column = None
        namespace = ORIGINAL_ID_NAMESPACE
        aliases = ("original_seg",)
        if "table_original" in sdata_obj.tables:
            table_key = "table_original"
    else:
        return
    register_segmentation_branch(
        sdata_obj,
        branch,
        points_key=points_key,
        assignment_column=assignment_column,
        background_column=background_column,
        shape_key=shape_key,
        table_key=table_key,
        id_namespace=namespace,
        legacy_aliases=aliases,
    )


def _write_table_element(sdata_obj: Any, table_key: str) -> None:
    """Write one table element to disk with version-compatible fallback."""
    if hasattr(sdata_obj, "write_element"):
        sdata_obj.write_element(table_key, overwrite=False)
        return
    logger.warning(
        "SpatialData object has no write_element(); writing full zarr fallback."
    )


def run_per_shape_assignment_for_dataset(
    dataset_name: str,
    latest_path: Path | str,
    *,
    force_rerun: bool = False,
    chunk_rows: int = 750_000,
    status_every_chunks: int = 5,
    table_prefix: str = "table_",
    memory_check_every_chunks: int = 5,
    max_ram_gb: float = 600.0,
    warn_ram_gb: float = 560.0,
) -> list[dict[str, Any]]:
    """Compute and persist per-shape assignment tables for one enriched dataset."""
    latest_path = Path(latest_path)
    dataset_name = str(dataset_name).upper()
    log_status(
        f"[{dataset_name}] Loading enriched zarr for per-shape assignment: "
        f"{latest_path}"
    )
    sdata_obj = sd.read_zarr(latest_path)

    if len(sdata_obj.points) == 0:
        raise RuntimeError(f"[{dataset_name}] No points found in {latest_path}")

    points_key = choose_primary_points_key(sdata_obj)
    if points_key is None:
        raise RuntimeError(f"[{dataset_name}] No primary transcript element found")
    points_obj = sdata_obj.points[points_key]
    stamp_merxen_schema(sdata_obj, primary_points_key=points_key)
    gene_list = build_gene_list_from_base_table(sdata_obj)
    shape_keys = list(sdata_obj.shapes.keys())
    log_status(f"[{dataset_name}] Points key='{points_key}', shape layers={shape_keys}")

    summaries: list[dict[str, Any]] = []
    wrote_any = False
    metadata_changed = True

    for shape_key in tqdm(
        shape_keys,
        desc=f"[{dataset_name}] shape tables",
        unit="shape",
    ):
        table_key = sanitize_table_key(shape_key, table_prefix=table_prefix)
        table_exists = table_key in sdata_obj.tables
        if table_exists and (not force_rerun):
            _register_assignment_branch(
                sdata_obj,
                points_key=points_key,
                shape_key=shape_key,
                table_key=table_key,
            )
            log_status(f"[{dataset_name}] Skipping existing table '{table_key}'")
            continue

        if table_exists and force_rerun:
            try:
                sdata_obj.delete_element_from_disk(table_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] delete_element_from_disk('%s') warning: %s",
                    dataset_name,
                    table_key,
                    exc,
                )
            with suppress(Exception):
                del sdata_obj.tables[table_key]

        shp, id_col = ensure_shape_has_cell_id(sdata_obj, shape_key)
        if (
            INSTANCE_ID_COLUMN not in sdata_obj.shapes[shape_key].columns
            or sdata_obj.shapes[shape_key].index.name != INSTANCE_ID_COLUMN
        ):
            try:
                transformations = get_transformation(
                    sdata_obj.shapes[shape_key],
                    get_all=True,
                )
            except (AssertionError, KeyError):
                transformations = None
            shp.attrs.pop("transform", None)
            normalized_shape = (
                ShapesModel.parse(shp, transformations=transformations)
                if transformations is not None
                else ShapesModel.parse(shp)
            )
            write_or_replace_element(
                sdata_obj,
                shape_key,
                "shapes",
                normalized_shape,
                overwrite=True,
            )
        source_table_key = shape_to_existing_table_source(shape_key)
        if source_table_key in sdata_obj.tables:
            try:
                table_obj = clone_table_for_region(
                    sdata_obj.tables[source_table_key],
                    shape_key,
                )
                sdata_obj.tables[table_key] = table_obj
                _write_table_element(sdata_obj, table_key)
                _register_assignment_branch(
                    sdata_obj,
                    points_key=points_key,
                    shape_key=shape_key,
                    table_key=table_key,
                )
                wrote_any = True
                summaries.append(
                    {
                        "dataset": dataset_name,
                        "shape_key": shape_key,
                        "table_key": table_key,
                        "mode": f"copied_from_{source_table_key}",
                        "n_cells": int(sdata_obj.tables[table_key].n_obs),
                        "n_genes": int(sdata_obj.tables[table_key].n_vars),
                        "n_points_input": np.nan,
                        "n_points_used": np.nan,
                        "n_points_assigned": np.nan,
                        "pct_assigned": np.nan,
                    }
                )
                log_status(
                    f"[{dataset_name}] Added '{table_key}' by cloning "
                    f"'{source_table_key}'"
                )
                continue
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"[{dataset_name}] Failed to clone table '{source_table_key}' "
                    f"for shape '{shape_key}'. Refusing to replace a source-backed "
                    "assignment table with a geometric spatial-join fallback."
                ) from exc

        table_obj, summary = compute_table_from_points_for_shape(
            dataset_name=dataset_name,
            points_obj=points_obj,
            shape_gdf=shp,
            shape_id_col=id_col,
            shape_key=shape_key,
            gene_list=gene_list,
            chunk_rows=chunk_rows,
            status_every_chunks=status_every_chunks,
            memory_check_every_chunks=memory_check_every_chunks,
            max_ram_gb=max_ram_gb,
            warn_ram_gb=warn_ram_gb,
        )
        sdata_obj.tables[table_key] = table_obj
        _write_table_element(sdata_obj, table_key)
        _register_assignment_branch(
            sdata_obj,
            points_key=points_key,
            shape_key=shape_key,
            table_key=table_key,
        )
        wrote_any = True

        summary["table_key"] = table_key
        summary["mode"] = "computed_sjoin"
        summaries.append(summary)
        log_status(
            f"[{dataset_name}] Added '{table_key}' | cells={summary['n_cells']:,} "
            f"genes={summary['n_genes']:,} assigned={summary['n_points_assigned']:,} "
            f"({summary['pct_assigned']:.2f}%)"
        )

    if wrote_any:
        if not hasattr(sdata_obj, "write_element"):
            write_spatialdata_zarr(sdata_obj, latest_path, overwrite=True)
        else:
            write_spatialdata_metadata(sdata_obj, write_attrs=True)
        force_release(note=f"after per-shape table writes ({dataset_name})")
        log_status(
            f"[{dataset_name}] Per-shape assignment tables written to: {latest_path}"
        )
    else:
        if metadata_changed and hasattr(sdata_obj, "write_metadata"):
            write_spatialdata_metadata(sdata_obj, write_attrs=True)
        force_release(note=f"after per-shape assignment no-op ({dataset_name})")
        log_status(f"[{dataset_name}] No per-shape assignment updates were needed")

    del sdata_obj
    return summaries
