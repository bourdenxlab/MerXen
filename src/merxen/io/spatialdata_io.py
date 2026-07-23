"""Zarr read/write and SpatialData V2 format conversion."""

# ruff: noqa: E402

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

os.environ["MPLCONFIGDIR"] = "./tmp/mpl"
os.environ["NUMBA_CACHE_DIR"] = "./tmp/numba"

import dask.dataframe as dd
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import get_transformation

from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    LEGACY_PROSEG_MASK_ASSIGNMENT_COLUMN,
    MERXEN_SCHEMA_ATTR,
    MERXEN_SCHEMA_VERSION,
    ORIGINAL_ID_NAMESPACE,
    PROSEG_ASSIGNMENT_COLUMN,
    PROSEG_GEOMETRY_ASSIGNMENT_COLUMN,
    PROSEG_ID_NAMESPACE,
    PROSEG_INTERNAL_ID_COLUMN,
    SOURCE_CELL_ID_COLUMN,
    TRANSCRIPT_ID_COLUMN,
    canonical_instance_series,
    choose_primary_points_key,
    register_segmentation_branch,
    stamp_merxen_schema,
    validate_merxen_schema,
    with_stable_transcript_ids,
)
from merxen.io.transcript_io import first_existing_col
from merxen.memory import force_release, log_status
from merxen.path_utils import remove_path

logger = logging.getLogger(__name__)

_ELEMENT_TYPE_ALIASES = {
    "image": "images",
    "images": "images",
    "label": "labels",
    "labels": "labels",
    "point": "points",
    "points": "points",
    "shape": "shapes",
    "shapes": "shapes",
    "table": "tables",
    "tables": "tables",
}


def write_spatialdata_zarr(
    sdata_obj: Any,
    path: Path,
    *,
    overwrite: bool | None = None,
) -> None:
    """Write a SpatialData object with optional overwrite semantics."""
    kwargs: dict[str, Any] = {}
    if overwrite is not None:
        kwargs["overwrite"] = overwrite
    sdata_obj.write(path, **kwargs)


def write_or_replace_element(
    sdata_obj: Any,
    key: str,
    element_type: str,
    value: Any,
    *,
    overwrite: bool = True,
) -> bool:
    """Add or replace one SpatialData element and persist only that element.

    The helper deliberately avoids deleting the on-disk element before writing a
    replacement. SpatialData's own ``write_element(..., overwrite=True)`` keeps
    that policy localized to the element writer and avoids the data-loss window
    from an explicit delete-then-write sequence.
    """
    container = _get_element_container(sdata_obj, element_type)
    element_key = str(key)
    exists = element_key in container
    if exists and not overwrite:
        return False

    try:
        container[element_key] = value
    except Exception:
        if not exists or not overwrite:
            raise
        # Some container implementations reject direct replacement. Remove only
        # the in-memory mapping entry; do not delete the existing on-disk data.
        del container[element_key]
        container[element_key] = value

    write_element = getattr(sdata_obj, "write_element", None)
    if callable(write_element):
        write_overwrite = exists
        try:
            write_element(element_key, overwrite=write_overwrite)
        except ValueError as exc:
            if not (overwrite and _can_retry_element_overwrite(exc)):
                raise
            logger.warning(
                "SpatialData write_element(overwrite=%s) failed for %s; "
                "retrying with a recoverable element backup.",
                write_overwrite,
                element_key,
            )
            if not _write_element_with_recoverable_backup(
                sdata_obj,
                element_key,
                element_type,
            ):
                _delete_element_from_disk_or_path(
                    sdata_obj,
                    element_key,
                    element_type,
                )
                write_element(element_key, overwrite=False)
    else:
        path = getattr(sdata_obj, "path", None)
        write = getattr(sdata_obj, "write", None)
        if path is not None and callable(write):
            write_spatialdata_zarr(sdata_obj, Path(path), overwrite=True)

    return True


def _write_element_with_recoverable_backup(
    sdata_obj: Any,
    element_key: str,
    element_type: str,
) -> bool:
    """Replace a local element while retaining the old store until success."""
    path = getattr(sdata_obj, "path", None)
    normalized = _ELEMENT_TYPE_ALIASES.get(str(element_type).lower())
    write_element = getattr(sdata_obj, "write_element", None)
    if path is None or normalized is None or not callable(write_element):
        return False

    element_path = Path(path) / normalized / element_key
    if not element_path.exists():
        return False
    backup_path = element_path.parent / (
        f".{element_key}.merxen-backup-{uuid.uuid4().hex}"
    )
    os.replace(element_path, backup_path)
    try:
        write_element(element_key, overwrite=False)
    except Exception:
        if element_path.exists() or element_path.is_symlink():
            remove_path(element_path)
        os.replace(backup_path, element_path)
        raise
    remove_path(backup_path)
    return True


def _delete_element_from_disk_or_path(
    sdata_obj: Any,
    element_key: str,
    element_type: str,
) -> None:
    """Delete an element store, including orphaned stores missing from metadata."""
    delete_element = getattr(sdata_obj, "delete_element_from_disk", None)
    if callable(delete_element):
        try:
            delete_element(element_key)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "delete_element_from_disk('%s') failed; trying path cleanup: %s",
                element_key,
                exc,
            )

    path = getattr(sdata_obj, "path", None)
    normalized = _ELEMENT_TYPE_ALIASES.get(str(element_type).lower())
    if path is None or normalized is None:
        raise RuntimeError(
            f"Cannot delete SpatialData element store for {element_key!r}; "
            "the object has no disk path."
        )
    element_path = Path(path) / normalized / element_key
    if not element_path.exists() and not element_path.is_symlink():
        raise FileNotFoundError(
            f"Cannot find on-disk SpatialData element store: {element_path}"
        )
    remove_path(element_path)


def write_spatialdata_metadata(
    sdata_obj: Any,
    *,
    write_attrs: bool = True,
    write_transformations: bool = False,
) -> None:
    """Persist SpatialData metadata without rewriting element data when possible."""
    if write_transformations:
        write_transformations_fn = getattr(sdata_obj, "write_transformations", None)
        if callable(write_transformations_fn):
            write_transformations_fn()

    write_metadata = getattr(sdata_obj, "write_metadata", None)
    if callable(write_metadata):
        write_metadata(write_attrs=write_attrs)


def _get_element_container(sdata_obj: Any, element_type: str) -> Any:
    """Return the SpatialData element mapping for a singular or plural type name."""
    normalized = _ELEMENT_TYPE_ALIASES.get(str(element_type).lower())
    if normalized is None:
        valid = ", ".join(sorted(_ELEMENT_TYPE_ALIASES))
        raise ValueError(f"Unknown SpatialData element type {element_type!r}: {valid}")
    return getattr(sdata_obj, normalized)


def _can_retry_element_overwrite(exc: ValueError) -> bool:
    """Return True for SpatialData's same-store overwrite refusal."""
    text = str(exc)
    return ("Cannot overwrite" in text and "target path" in text) or (
        "Zarr store already exists" in text
        and "currently in use by the current SpatialData object" in text
    )


def normalize_points_for_latest_write(
    points_obj: Any,
    points_key: str = "points",
) -> Any:
    """Normalize point-table dtypes so all Dask partitions share one pyarrow schema.

    Gene dictionaries can exceed int8 code range across partitions; this forces
    plain string types. Mixed integer/float/string identifiers are coerced to
    consistent types.

    Args:
        points_obj: A Dask or pandas DataFrame of transcript points.
        points_key: Name of the points element (for logging).

    Returns:
        The DataFrame with normalized column types.
    """
    if not hasattr(points_obj, "columns"):
        return points_obj

    df = points_obj
    cols = set(map(str, list(df.columns)))

    df = _preserve_observed_transcript_coordinates(df, cols)
    cols = set(map(str, list(df.columns)))

    # Gene columns: force to plain string to avoid categorical overflow
    for gene_col in ("gene", "feature_name", "target"):
        if gene_col in cols:
            try:
                df[gene_col] = df[gene_col].astype("string")
            except Exception:  # noqa: BLE001
                df[gene_col] = df[gene_col].astype(str)

    # Integer identifier columns: prefer unsigned ints, fall back to string.
    # ProSeg's assignment column is nullable; NULL means background/unassigned,
    # while 0 can be a valid zero-based ProSeg cell id.
    int_casts: dict[str, str] = {"cell": "uint32"}
    for c, target_dtype in int_casts.items():
        if c in cols:
            try:
                df[c] = df[c].astype("float64").fillna(0).astype(target_dtype)
            except Exception:  # noqa: BLE001
                try:
                    df[c] = df[c].fillna(0).astype(target_dtype)
                except Exception:  # noqa: BLE001
                    df[c] = df[c].astype("string")

    if TRANSCRIPT_ID_COLUMN in cols:
        try:
            df[TRANSCRIPT_ID_COLUMN] = (
                df[TRANSCRIPT_ID_COLUMN].astype("float64").astype("UInt64")
            )
        except Exception:  # noqa: BLE001
            df[TRANSCRIPT_ID_COLUMN] = df[TRANSCRIPT_ID_COLUMN].astype("string")

    if "assignment" in cols:
        try:
            df["assignment"] = df["assignment"].astype("float64").astype("UInt32")
        except Exception:  # noqa: BLE001
            try:
                numeric_assignment = pd.to_numeric(
                    df["assignment"],
                    errors="coerce",
                )
                df["assignment"] = numeric_assignment.astype("UInt32")
            except Exception:  # noqa: BLE001
                df["assignment"] = df["assignment"].astype("string")

    # cell_id as string for downstream assignment logic compatibility
    if "cell_id" in cols:
        try:
            df["cell_id"] = df["cell_id"].astype("string")
        except Exception:  # noqa: BLE001
            df["cell_id"] = df["cell_id"].astype(str)

    log_status(f"Normalized points schema for '{points_key}' (columns={len(cols)})")
    return df


def _preserve_observed_transcript_coordinates(df: Any, cols: set[str]) -> Any:
    """Make observed transcript coordinates canonical when ProSeg moved them.

    ProSeg writes inferred/repositioned transcript coordinates to ``x/y/z`` and
    the physical detected coordinates to ``observed_x/observed_y/observed_z``.
    MerXen's downstream code expects ``x/y`` to be physical transcript
    positions, so keep the ProSeg coordinates under explicit names.
    """
    for coord in ("x", "y", "z"):
        observed_col = f"observed_{coord}"
        moved_col = f"proseg_moved_{coord}"
        if coord not in cols or observed_col not in cols:
            continue
        if moved_col not in cols:
            df[moved_col] = df[coord]
            cols.add(moved_col)
        df[coord] = df[observed_col]
    return df


def _set_known_feature_categories(points_obj: Any, feature_key: str) -> Any:
    """Use a finite, known categorical gene vocabulary across Dask partitions."""
    if not isinstance(points_obj, dd.DataFrame):
        points_obj[feature_key] = points_obj[feature_key].astype("category")
        return points_obj
    feature_values = (
        points_obj[feature_key]
        .astype("string")
        .dropna()
        .drop_duplicates()
        .compute()
        .astype(str)
        .sort_values()
        .tolist()
    )
    dtype = pd.CategoricalDtype(categories=feature_values)
    points_obj[feature_key] = points_obj[feature_key].astype("string").astype(dtype)
    return points_obj


def _positive_label(value: Any) -> int | None:
    """Parse a positive mask label from a numeric or prefixed source ID."""
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = int(float(text))
    except ValueError:
        match = re.search(r"([0-9]+)$", text)
        if match is None:
            return None
        numeric = int(match.group(1))
    return numeric if numeric > 0 else None


def _stable_source_instance_mapping(values: Any) -> dict[str, int]:
    """Map opaque source IDs to deterministic positive operational IDs."""
    source = pd.Series(values, copy=False).astype(str)
    if bool(source.duplicated().any()):
        raise ValueError("Source segmentation contains duplicate cell identifiers")
    parsed: list[int] = []
    all_positive_integers = True
    for value in source:
        try:
            number = int(value)
        except ValueError:
            all_positive_integers = False
            break
        if number <= 0 or number > np.iinfo(np.uint64).max:
            all_positive_integers = False
            break
        parsed.append(number)
    if all_positive_integers and len(set(parsed)) == len(parsed):
        return dict(zip(source.tolist(), parsed, strict=True))
    return {
        source_id: index + 1 for index, source_id in enumerate(sorted(source.tolist()))
    }


def _canonicalize_source_points_partition(
    partition: pd.DataFrame,
    *,
    source_assignment_column: str | None,
    mapping: dict[str, int],
) -> pd.DataFrame:
    """Add canonical original-segmentation assignments to one points partition."""
    result = partition.copy()
    if source_assignment_column is None:
        return result
    source = result[source_assignment_column].astype("string")
    normalized = source.str.strip()
    unassigned = normalized.isna() | normalized.str.lower().isin(
        {"", "0", "-1", "nan", "none", "<na>", "unassigned"}
    )
    mapped = normalized.map(mapping)
    unknown = ~unassigned & mapped.isna()
    if bool(unknown.any()):
        examples = normalized.loc[unknown].head(5).tolist()
        raise ValueError(
            f"Transcript assignments reference unknown source cells: {examples}"
        )
    result[SOURCE_CELL_ID_COLUMN] = normalized.mask(unassigned, pd.NA)
    result["original_assignment"] = mapped.astype("UInt64")
    if source_assignment_column not in {
        SOURCE_CELL_ID_COLUMN,
        "original_assignment",
    }:
        result = result.drop(columns=[source_assignment_column])
    return result


def prepare_source_spatialdata_contract(
    sdata_obj: Any,
    *,
    platform: str,
) -> None:
    """Canonicalize an instrument-reader SpatialData object before first write."""
    primary_points = choose_primary_points_key(sdata_obj)
    stamp_merxen_schema(
        sdata_obj,
        primary_points_key=primary_points,
        platform=platform,
    )

    tables = getattr(sdata_obj, "tables", {})
    points = getattr(sdata_obj, "points", {})
    table_key = "table" if "table" in tables else None
    table_obj = tables.get(table_key) if table_key is not None else None
    if table_obj is None:
        for points_key in list(points):
            points[points_key] = with_stable_transcript_ids(
                points[points_key],
                preserve_existing=True,
            )
        return

    attrs = dict(table_obj.uns.get("spatialdata_attrs", {}))
    old_instance_key = attrs.get("instance_key")
    if old_instance_key not in table_obj.obs.columns:
        old_instance_key = first_existing_col(
            table_obj.obs,
            ["cell_id", "cell", "EntityID"],
        )
    source_ids = (
        table_obj.obs[old_instance_key].astype(str)
        if old_instance_key is not None
        else pd.Series(table_obj.obs_names.astype(str), index=table_obj.obs.index)
    )
    mapping = _stable_source_instance_mapping(source_ids)
    instance_ids = source_ids.map(mapping).astype("uint64")

    region_key = str(attrs.get("region_key", "region"))
    regions = attrs.get("region")
    if isinstance(regions, str):
        region_names = [regions]
    elif regions is None:
        region_names = table_obj.obs[region_key].astype(str).unique().tolist()
    else:
        region_names = [str(value) for value in regions]

    for region_name in region_names:
        if region_name not in sdata_obj.shapes:
            continue
        shape = sdata_obj.shapes[region_name]
        shape_source_col = first_existing_col(
            shape,
            [
                SOURCE_CELL_ID_COLUMN,
                str(old_instance_key) if old_instance_key is not None else "",
                "cell_id",
                "EntityID",
            ],
        )
        shape_source = (
            shape[shape_source_col].astype(str)
            if shape_source_col is not None
            else pd.Series(shape.index.astype(str), index=shape.index)
        )
        shape_ids = shape_source.map(mapping)
        if bool(shape_ids.isna().any()):
            raise ValueError(
                f"Shape {region_name!r} contains IDs absent from its cell table"
            )
        gdf = shape.copy()
        gdf[SOURCE_CELL_ID_COLUMN] = shape_source.to_numpy(dtype=object)
        gdf[INSTANCE_ID_COLUMN] = shape_ids.to_numpy(dtype=np.uint64)
        if shape_source_col not in {
            None,
            SOURCE_CELL_ID_COLUMN,
            INSTANCE_ID_COLUMN,
        }:
            gdf = gdf.drop(columns=[shape_source_col])
        gdf.index = pd.Index(
            gdf[INSTANCE_ID_COLUMN],
            dtype="uint64",
            name=INSTANCE_ID_COLUMN,
        )
        transformations = get_transformation(shape, get_all=True)
        gdf.attrs.pop("transform", None)
        sdata_obj.shapes[region_name] = ShapesModel.parse(
            gdf,
            transformations=transformations,
        )

    table = table_obj.copy()
    table.obs[SOURCE_CELL_ID_COLUMN] = source_ids.to_numpy(dtype=object)
    table.obs[INSTANCE_ID_COLUMN] = instance_ids.to_numpy(dtype=np.uint64)
    if old_instance_key not in {
        None,
        SOURCE_CELL_ID_COLUMN,
        INSTANCE_ID_COLUMN,
    }:
        del table.obs[old_instance_key]
    table.obs_names = table.obs[INSTANCE_ID_COLUMN].astype(str).to_numpy()
    table.uns.pop("spatialdata_attrs", None)
    sdata_obj.tables[table_key] = TableModel.parse(
        table,
        region=region_names[0] if len(region_names) == 1 else region_names,
        region_key=region_key,
        instance_key=INSTANCE_ID_COLUMN,
    )

    for points_key in list(sdata_obj.points):
        points_obj = sdata_obj.points[points_key]
        transformations = get_transformation(points_obj, get_all=True)
        point_attrs = dict(points_obj.attrs.get("spatialdata_attrs", {}))
        source_assignment_col = point_attrs.get("instance_key")
        if source_assignment_col not in points_obj.columns:
            source_assignment_col = first_existing_col(
                points_obj,
                ["cell_id", "EntityID"],
            )
        if isinstance(points_obj, dd.DataFrame):
            meta = _canonicalize_source_points_partition(
                points_obj._meta,
                source_assignment_column=source_assignment_col,
                mapping=mapping,
            )
            points_obj = points_obj.map_partitions(
                _canonicalize_source_points_partition,
                source_assignment_column=source_assignment_col,
                mapping=mapping,
                meta=meta,
            )
        else:
            points_obj = _canonicalize_source_points_partition(
                points_obj,
                source_assignment_column=source_assignment_col,
                mapping=mapping,
            )
        points_obj = with_stable_transcript_ids(
            points_obj,
            preserve_existing=True,
        ).reset_index(drop=True)
        x_col = first_existing_col(
            points_obj,
            ["x", "global_x", "x_location"],
        )
        y_col = first_existing_col(
            points_obj,
            ["y", "global_y", "y_location"],
        )
        z_col = first_existing_col(
            points_obj,
            ["z", "global_z", "z_location"],
        )
        gene_col = first_existing_col(
            points_obj,
            ["gene", "feature_name", "target"],
        )
        if x_col is None or y_col is None or gene_col is None:
            raise KeyError("Could not resolve source transcript coordinates and gene")
        if isinstance(points_obj, dd.DataFrame):
            points_obj = _set_known_feature_categories(points_obj, gene_col)
        coordinates = {"x": x_col, "y": y_col}
        if z_col is not None:
            coordinates["z"] = z_col
        parse_kwargs: dict[str, Any] = {
            "coordinates": coordinates,
            "feature_key": gene_col,
            "transformations": transformations,
            "sort": True,
        }
        if "original_assignment" in points_obj.columns:
            parse_kwargs["instance_key"] = "original_assignment"
        sdata_obj.points[points_key] = PointsModel.parse(points_obj, **parse_kwargs)

    if primary_points is not None and region_names:
        register_segmentation_branch(
            sdata_obj,
            "original",
            points_key=primary_points,
            assignment_column=(
                "original_assignment"
                if "original_assignment" in sdata_obj.points[primary_points].columns
                else None
            ),
            shape_key=region_names[0],
            table_key=table_key,
            id_namespace=ORIGINAL_ID_NAMESPACE,
            legacy_aliases=("original_seg",),
        )
        validate_merxen_schema(sdata_obj, deep=False)


def _proseg_identifier_mapping(
    sdata_obj: Any,
) -> tuple[dict[int, int], dict[int, str], str]:
    """Build a deterministic ProSeg-internal to canonical instance mapping."""
    candidate_tables = [
        key for key in ("table", "table_MOSAIK_proseg") if key in sdata_obj.tables
    ]
    table_key = next(
        (
            key
            for key in candidate_tables
            if (
                "original_cell_id" in sdata_obj.tables[key].obs.columns
                or SOURCE_CELL_ID_COLUMN in sdata_obj.tables[key].obs.columns
            )
        ),
        candidate_tables[0] if candidate_tables else None,
    )
    if table_key is None:
        raise KeyError("Could not find the ProSeg cell table used for ID mapping")

    obs = sdata_obj.tables[table_key].obs
    internal_col = first_existing_col(
        obs,
        [PROSEG_INTERNAL_ID_COLUMN, "cell", "cell_id"],
    )
    if internal_col is None:
        internal = pd.Series(
            np.arange(len(obs), dtype=np.uint64),
            index=obs.index,
        )
    else:
        internal = pd.to_numeric(obs[internal_col], errors="raise").astype("uint64")
    source_col = first_existing_col(
        obs,
        [SOURCE_CELL_ID_COLUMN, "original_cell_id"],
    )
    source = (
        obs[source_col].astype(str)
        if source_col is not None
        else pd.Series(internal.astype(str), index=obs.index)
    )

    provisional = [_positive_label(value) for value in source]
    valid_values = [value for value in provisional if value is not None]
    next_allocated = max(valid_values, default=0) + 1
    used: set[int] = set()
    mapping: dict[int, int] = {}
    source_ids: dict[int, str] = {}
    rows = sorted(
        zip(internal.tolist(), source.tolist(), provisional, strict=True),
        key=lambda item: int(item[0]),
    )
    for internal_id, source_id, candidate in rows:
        canonical = candidate
        if canonical is None or canonical in used:
            while next_allocated in used:
                next_allocated += 1
            canonical = next_allocated
            next_allocated += 1
        used.add(int(canonical))
        mapping[int(internal_id)] = int(canonical)
        source_ids[int(internal_id)] = str(source_id)
    return mapping, source_ids, table_key


def _map_nullable_ids(values: pd.Series, mapping: dict[int, int]) -> pd.Series:
    """Map nullable zero-based internal IDs to positive canonical IDs."""
    numeric = pd.to_numeric(values, errors="coerce")
    mapped = numeric.map(mapping)
    missing = numeric.notna() & mapped.isna()
    if bool(missing.any()):
        examples = numeric.loc[missing].head(5).tolist()
        raise ValueError(f"Assignments reference unknown ProSeg cells: {examples}")
    return mapped.astype("UInt64")


def _canonicalize_proseg_points_partition(
    partition: pd.DataFrame,
    *,
    mapping: dict[int, int],
    quality_column_alias: str | None,
) -> pd.DataFrame:
    """Canonicalize assignment fields in one transcript partition."""
    result = partition.copy()
    if PROSEG_ASSIGNMENT_COLUMN in result.columns:
        if PROSEG_INTERNAL_ID_COLUMN in result.columns:
            internal = pd.to_numeric(
                result[PROSEG_INTERNAL_ID_COLUMN],
                errors="coerce",
            ).astype("UInt64")
        else:
            internal = pd.to_numeric(
                result[PROSEG_ASSIGNMENT_COLUMN],
                errors="coerce",
            ).astype("UInt64")
        result[PROSEG_INTERNAL_ID_COLUMN] = internal
        result[PROSEG_ASSIGNMENT_COLUMN] = _map_nullable_ids(internal, mapping)

    if LEGACY_PROSEG_MASK_ASSIGNMENT_COLUMN in result.columns:
        result[PROSEG_GEOMETRY_ASSIGNMENT_COLUMN] = _map_nullable_ids(
            result[LEGACY_PROSEG_MASK_ASSIGNMENT_COLUMN],
            mapping,
        )
        result = result.drop(columns=[LEGACY_PROSEG_MASK_ASSIGNMENT_COLUMN])
    elif PROSEG_GEOMETRY_ASSIGNMENT_COLUMN in result.columns:
        result[PROSEG_GEOMETRY_ASSIGNMENT_COLUMN] = canonical_instance_series(
            result[PROSEG_GEOMETRY_ASSIGNMENT_COLUMN],
            field_name=PROSEG_GEOMETRY_ASSIGNMENT_COLUMN,
            allow_null=True,
        )

    if "hybrid_assignment" in result.columns:
        result["hybrid_assignment"] = canonical_instance_series(
            result["hybrid_assignment"],
            field_name="hybrid_assignment",
            allow_null=True,
        )
    if quality_column_alias is not None and "qv" in result.columns:
        result[quality_column_alias] = pd.to_numeric(
            result["qv"],
            errors="coerce",
        ).astype("float32")
    if "hybrid_assignment_source" in result.columns:
        result["hybrid_assignment_source"] = result["hybrid_assignment_source"].astype(
            "category"
        )
    return result


def _canonicalize_proseg_points(
    points_obj: Any,
    *,
    mapping: dict[int, int],
    quality_column_alias: str | None,
) -> Any:
    """Canonicalize and reparse one ProSeg transcript points element."""
    transformations = get_transformation(points_obj, get_all=True)
    points_obj = normalize_points_for_latest_write(points_obj)
    if isinstance(points_obj, dd.DataFrame):
        meta = _canonicalize_proseg_points_partition(
            points_obj._meta,
            mapping=mapping,
            quality_column_alias=quality_column_alias,
        )
        points_obj = points_obj.map_partitions(
            _canonicalize_proseg_points_partition,
            mapping=mapping,
            quality_column_alias=quality_column_alias,
            meta=meta,
        )
    else:
        points_obj = _canonicalize_proseg_points_partition(
            points_obj,
            mapping=mapping,
            quality_column_alias=quality_column_alias,
        )
    points_obj = with_stable_transcript_ids(points_obj)
    points_obj = points_obj.reset_index(drop=True)

    x_col = first_existing_col(
        points_obj,
        ["x", "observed_x", "x_micron", "global_x"],
    )
    y_col = first_existing_col(
        points_obj,
        ["y", "observed_y", "y_micron", "global_y"],
    )
    z_col = first_existing_col(
        points_obj,
        ["z", "observed_z", "z_micron", "global_z"],
    )
    gene_col = first_existing_col(points_obj, ["gene", "feature_name", "target"])
    if x_col is None or y_col is None or gene_col is None:
        raise KeyError(
            "Could not resolve coordinates and gene column for ProSeg points"
        )
    if isinstance(points_obj, dd.DataFrame):
        points_obj = _set_known_feature_categories(points_obj, gene_col)

    coordinates = {"x": x_col, "y": y_col}
    if z_col is not None:
        coordinates["z"] = z_col
    parse_kwargs: dict[str, Any] = {
        "coordinates": coordinates,
        "feature_key": gene_col,
        "transformations": transformations,
        "sort": True,
    }
    if PROSEG_ASSIGNMENT_COLUMN in points_obj.columns:
        parse_kwargs["instance_key"] = PROSEG_ASSIGNMENT_COLUMN
    return PointsModel.parse(points_obj, **parse_kwargs)


def _canonicalize_proseg_shape(
    shape_obj: Any,
    *,
    mapping: dict[int, int],
    source_ids: dict[int, str],
) -> Any:
    """Canonicalize a ProSeg-derived shape while preserving internal IDs."""
    gdf = shape_obj.copy()
    internal_col = first_existing_col(
        gdf,
        [PROSEG_INTERNAL_ID_COLUMN, "cell", "cell_id"],
    )
    internal = (
        pd.to_numeric(gdf[internal_col], errors="raise").astype("uint64")
        if internal_col is not None
        else pd.Series(gdf.index, index=gdf.index).astype("uint64")
    )
    canonical = internal.map(mapping)
    if bool(canonical.isna().any()):
        raise ValueError("ProSeg shape contains cells absent from the ProSeg table")
    gdf[PROSEG_INTERNAL_ID_COLUMN] = internal.to_numpy(dtype=np.uint64)
    gdf[SOURCE_CELL_ID_COLUMN] = [source_ids[int(value)] for value in internal.tolist()]
    gdf[INSTANCE_ID_COLUMN] = canonical.to_numpy(dtype=np.uint64)
    gdf = gdf.drop(
        columns=[column for column in ("cell", "cell_id") if column in gdf.columns]
    )
    gdf.index = pd.Index(
        gdf[INSTANCE_ID_COLUMN].to_numpy(dtype=np.uint64),
        dtype="uint64",
        name=INSTANCE_ID_COLUMN,
    )
    transformations = get_transformation(shape_obj, get_all=True)
    gdf.attrs.pop("transform", None)
    return ShapesModel.parse(gdf, transformations=transformations)


def _direct_shape_instance_ids(shape_obj: Any) -> Any:
    """Canonicalize Cellpose/hybrid shapes whose IDs already encode mask labels."""
    gdf = shape_obj.copy()
    source_col = first_existing_col(
        gdf,
        [SOURCE_CELL_ID_COLUMN, INSTANCE_ID_COLUMN, "cell_id", "cellpose_label"],
    )
    source = (
        gdf[source_col].astype(str)
        if source_col is not None
        else pd.Series(gdf.index.astype(str), index=gdf.index)
    )
    parsed = [_positive_label(value) for value in source]
    if any(value is None for value in parsed):
        raise ValueError("Could not parse positive mask labels from shape identifiers")
    canonical = np.asarray(parsed, dtype=np.uint64)
    if len(np.unique(canonical)) != len(canonical):
        raise ValueError("Shape identifiers are not unique")
    gdf[SOURCE_CELL_ID_COLUMN] = source.to_numpy(dtype=object)
    gdf[INSTANCE_ID_COLUMN] = canonical
    gdf = gdf.drop(
        columns=[column for column in ("cell", "cell_id") if column in gdf.columns]
    )
    gdf.index = pd.Index(
        canonical,
        dtype="uint64",
        name=INSTANCE_ID_COLUMN,
    )
    transformations = get_transformation(shape_obj, get_all=True)
    gdf.attrs.pop("transform", None)
    return ShapesModel.parse(gdf, transformations=transformations)


def _canonicalize_table(
    table_obj: Any,
    *,
    mode: str,
    mapping: dict[int, int],
    source_ids: dict[int, str],
) -> Any:
    """Canonicalize the instance key for one derived cell table."""
    table = table_obj.copy()
    attrs = dict(table.uns.get("spatialdata_attrs", {}))
    old_instance_key = attrs.get("instance_key")
    if mode == "proseg":
        internal_col = first_existing_col(
            table.obs,
            [PROSEG_INTERNAL_ID_COLUMN, "cell", "cell_id"],
        )
        internal = (
            pd.to_numeric(table.obs[internal_col], errors="raise").astype("uint64")
            if internal_col is not None
            else pd.Series(
                np.arange(len(table.obs), dtype=np.uint64),
                index=table.obs.index,
            )
        )
        canonical = internal.map(mapping)
        if bool(canonical.isna().any()):
            raise ValueError("ProSeg table contains cells absent from the ID mapping")
        table.obs[PROSEG_INTERNAL_ID_COLUMN] = internal.to_numpy(dtype=np.uint64)
        table.obs[SOURCE_CELL_ID_COLUMN] = [
            source_ids[int(value)] for value in internal.tolist()
        ]
        table.obs[INSTANCE_ID_COLUMN] = canonical.to_numpy(dtype=np.uint64)
    else:
        source_candidates = [SOURCE_CELL_ID_COLUMN, INSTANCE_ID_COLUMN]
        if isinstance(old_instance_key, str):
            source_candidates.append(old_instance_key)
        source_candidates.append("cell_id")
        source_col = first_existing_col(
            table.obs,
            source_candidates,
        )
        source = (
            table.obs[source_col].astype(str)
            if source_col is not None
            else pd.Series(table.obs_names.astype(str), index=table.obs.index)
        )
        parsed = [_positive_label(value) for value in source]
        if any(value is None for value in parsed):
            raise ValueError("Could not parse positive table instance identifiers")
        table.obs[SOURCE_CELL_ID_COLUMN] = source.to_numpy(dtype=object)
        table.obs[INSTANCE_ID_COLUMN] = np.asarray(parsed, dtype=np.uint64)

    for column in ("cell", "cell_id"):
        if column in table.obs.columns and column != INSTANCE_ID_COLUMN:
            del table.obs[column]
    table.obs_names = table.obs[INSTANCE_ID_COLUMN].astype(str).to_numpy()
    region_key = str(attrs.get("region_key", "region"))
    region = attrs.get("region")
    if region is None:
        if region_key not in table.obs.columns:
            raise ValueError("Table has no SpatialData region metadata")
        values = table.obs[region_key].astype(str).unique().tolist()
        region = values[0] if len(values) == 1 else values
    table.uns.pop("spatialdata_attrs", None)
    return TableModel.parse(
        table,
        region=region,
        region_key=region_key,
        instance_key=INSTANCE_ID_COLUMN,
    )


def _canonicalize_original_region(
    sdata_obj: Any,
) -> tuple[str, str] | None:
    """Canonicalize the platform-original table and its annotated shape."""
    table_key = "table_original"
    if table_key not in sdata_obj.tables:
        return None
    table_obj = sdata_obj.tables[table_key]
    attrs = dict(table_obj.uns.get("spatialdata_attrs", {}))
    region = attrs.get("region")
    if isinstance(region, str):
        shape_key = region
    elif region is not None and len(region) == 1:
        shape_key = str(region[0])
    else:
        return None
    if shape_key not in sdata_obj.shapes:
        return None

    old_instance_key = attrs.get("instance_key")
    if old_instance_key not in table_obj.obs.columns:
        old_instance_key = first_existing_col(
            table_obj.obs,
            [SOURCE_CELL_ID_COLUMN, "cell_id", "cell", "EntityID"],
        )
    table_source = (
        table_obj.obs[old_instance_key].astype(str)
        if old_instance_key is not None
        else pd.Series(table_obj.obs_names.astype(str), index=table_obj.obs.index)
    )
    mapping = _stable_source_instance_mapping(table_source)
    table_ids = table_source.map(mapping).astype("uint64")

    shape_obj = sdata_obj.shapes[shape_key]
    shape_source_col = first_existing_col(
        shape_obj,
        [
            SOURCE_CELL_ID_COLUMN,
            str(old_instance_key) if old_instance_key is not None else "",
            "cell_id",
            "cell",
            "EntityID",
        ],
    )
    shape_source = (
        shape_obj[shape_source_col].astype(str)
        if shape_source_col is not None
        else pd.Series(shape_obj.index.astype(str), index=shape_obj.index)
    )
    shape_ids = shape_source.map(mapping)
    if bool(shape_ids.isna().any()):
        raise ValueError("Original shape IDs do not match table_original")
    gdf = shape_obj.copy()
    gdf[SOURCE_CELL_ID_COLUMN] = shape_source.to_numpy(dtype=object)
    gdf[INSTANCE_ID_COLUMN] = shape_ids.to_numpy(dtype=np.uint64)
    if shape_source_col not in {
        None,
        SOURCE_CELL_ID_COLUMN,
        INSTANCE_ID_COLUMN,
    }:
        gdf = gdf.drop(columns=[shape_source_col])
    gdf.index = pd.Index(
        gdf[INSTANCE_ID_COLUMN],
        dtype="uint64",
        name=INSTANCE_ID_COLUMN,
    )
    transformations = get_transformation(shape_obj, get_all=True)
    gdf.attrs.pop("transform", None)
    sdata_obj.shapes[shape_key] = ShapesModel.parse(
        gdf,
        transformations=transformations,
    )

    table = table_obj.copy()
    table.obs[SOURCE_CELL_ID_COLUMN] = table_source.to_numpy(dtype=object)
    table.obs[INSTANCE_ID_COLUMN] = table_ids.to_numpy(dtype=np.uint64)
    if old_instance_key not in {
        None,
        SOURCE_CELL_ID_COLUMN,
        INSTANCE_ID_COLUMN,
    }:
        del table.obs[old_instance_key]
    table.obs_names = table.obs[INSTANCE_ID_COLUMN].astype(str).to_numpy()
    region_key = str(attrs.get("region_key", "region"))
    table.uns.pop("spatialdata_attrs", None)
    sdata_obj.tables[table_key] = TableModel.parse(
        table,
        region=shape_key,
        region_key=region_key,
        instance_key=INSTANCE_ID_COLUMN,
    )
    return shape_key, table_key


def upgrade_spatialdata_contract_in_memory(
    sdata_obj: Any,
    *,
    quality_column_alias: str | None = None,
    platform: str | None = None,
) -> bool:
    """Upgrade an in-memory ProSeg SpatialData object to the current contract."""
    current = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    if current.get("schema_version") == MERXEN_SCHEMA_VERSION:
        return False

    mapping, source_ids, base_table_key = _proseg_identifier_mapping(sdata_obj)
    primary_points = choose_primary_points_key(sdata_obj)
    for points_key in list(sdata_obj.points):
        sdata_obj.points[points_key] = _canonicalize_proseg_points(
            sdata_obj.points[points_key],
            mapping=mapping,
            quality_column_alias=quality_column_alias,
        )

    proseg_shape_keys = [
        str(key)
        for key in list(sdata_obj.shapes)
        if (
            str(key) in {"cell_boundaries", "cell_boundaries_refined"}
            or ("proseg" in str(key).lower() and "hybrid" not in str(key).lower())
        )
    ]
    for shape_key in proseg_shape_keys:
        sdata_obj.shapes[shape_key] = _canonicalize_proseg_shape(
            sdata_obj.shapes[shape_key],
            mapping=mapping,
            source_ids=source_ids,
        )
    for shape_key in list(sdata_obj.shapes):
        lower = str(shape_key).lower()
        if "hybrid" in lower or "cellpose" in lower:
            sdata_obj.shapes[shape_key] = _direct_shape_instance_ids(
                sdata_obj.shapes[shape_key]
            )

    renamed_tables: dict[str, Any] = {}
    for table_key in list(sdata_obj.tables):
        lower = str(table_key).lower()
        if (
            table_key == base_table_key
            or (
                "proseg" in lower
                and "hybrid" not in lower
                and "mask_assignment" not in lower
                and "geometry_assignment" not in lower
            )
            or "proseg_mask_assignment" in lower
            or "proseg_geometry_assignment" in lower
        ):
            mode = "proseg"
        elif "hybrid" in lower or "cellpose" in lower:
            mode = "direct"
        else:
            continue
        normalized_key = str(table_key).replace(
            "proseg_mask_assignment",
            "proseg_geometry_assignment",
        )
        renamed_tables[normalized_key] = _canonicalize_table(
            sdata_obj.tables[table_key],
            mode=mode,
            mapping=mapping,
            source_ids=source_ids,
        )
        if normalized_key != table_key:
            del sdata_obj.tables[table_key]
    for table_key, table in renamed_tables.items():
        sdata_obj.tables[table_key] = table

    original_region = _canonicalize_original_region(sdata_obj)
    stamp_merxen_schema(
        sdata_obj,
        primary_points_key=primary_points,
        platform=platform,
    )
    proseg_shape = (
        "MOSAIK_proseg" if "MOSAIK_proseg" in sdata_obj.shapes else proseg_shape_keys[0]
    )
    proseg_table = (
        "table_MOSAIK_proseg"
        if "table_MOSAIK_proseg" in sdata_obj.tables
        else base_table_key
    )
    if primary_points is not None:
        register_segmentation_branch(
            sdata_obj,
            "proseg",
            points_key=primary_points,
            assignment_column=PROSEG_ASSIGNMENT_COLUMN,
            background_column="background",
            shape_key=proseg_shape,
            table_key=proseg_table,
            id_namespace=PROSEG_ID_NAMESPACE,
            legacy_aliases=("reseg",),
        )
        geometry_table = "table_MOSAIK_proseg_geometry_assignment"
        if (
            PROSEG_GEOMETRY_ASSIGNMENT_COLUMN
            in sdata_obj.points[primary_points].columns
        ):
            register_segmentation_branch(
                sdata_obj,
                "proseg_geometry_assignment",
                points_key=primary_points,
                assignment_column=PROSEG_GEOMETRY_ASSIGNMENT_COLUMN,
                shape_key=proseg_shape,
                table_key=(
                    geometry_table if geometry_table in sdata_obj.tables else None
                ),
                id_namespace=PROSEG_ID_NAMESPACE,
            )
        if (
            "MOSAIK_cellpose" in sdata_obj.shapes
            and "table_MOSAIK_cellpose" in sdata_obj.tables
        ):
            register_segmentation_branch(
                sdata_obj,
                "cellpose",
                points_key=primary_points,
                assignment_column=None,
                shape_key="MOSAIK_cellpose",
                table_key="table_MOSAIK_cellpose",
                id_namespace=PROSEG_ID_NAMESPACE,
                legacy_aliases=("proseg_mask", "cellpose_mask"),
            )
        if (
            "MOSAIK_proseg_hybrid" in sdata_obj.shapes
            and "table_MOSAIK_proseg_hybrid" in sdata_obj.tables
            and "hybrid_assignment" in sdata_obj.points[primary_points].columns
        ):
            register_segmentation_branch(
                sdata_obj,
                "proseg_hybrid",
                points_key=primary_points,
                assignment_column="hybrid_assignment",
                background_column="hybrid_background",
                assignment_source_column="hybrid_assignment_source",
                shape_key="MOSAIK_proseg_hybrid",
                table_key="table_MOSAIK_proseg_hybrid",
                id_namespace=PROSEG_ID_NAMESPACE,
            )
        if original_region is not None:
            original_shape, original_table = original_region
            register_segmentation_branch(
                sdata_obj,
                "original",
                points_key=primary_points,
                assignment_column=None,
                shape_key=original_shape,
                table_key=original_table,
                id_namespace=ORIGINAL_ID_NAMESPACE,
                legacy_aliases=("original_seg",),
            )
        schema = dict(sdata_obj.attrs.get(MERXEN_SCHEMA_ATTR, {}))
        native_branches = dict(schema.get("segmentations", {}))
        suffix = "_aligned_nonrigid"
        for branch, entry in native_branches.items():
            if entry.get("coordinate_variant_of") is not None:
                continue
            aligned_points = f"{entry['points']}{suffix}"
            aligned_shape = f"{entry['shape']}{suffix}"
            if (
                aligned_points not in sdata_obj.points
                or aligned_shape not in sdata_obj.shapes
            ):
                continue
            assignment_column = entry.get("assignment_column")
            if (
                assignment_column is not None
                and assignment_column not in sdata_obj.points[aligned_points].columns
            ):
                continue
            native_table = entry.get("table")
            aligned_table = (
                f"{native_table}{suffix}"
                if native_table is not None
                and f"{native_table}{suffix}" in sdata_obj.tables
                else None
            )
            register_segmentation_branch(
                sdata_obj,
                f"{branch}{suffix}",
                points_key=aligned_points,
                assignment_column=assignment_column,
                background_column=entry.get("background_column"),
                assignment_source_column=entry.get("assignment_source_column"),
                shape_key=aligned_shape,
                table_key=aligned_table,
                instance_key=str(entry.get("instance_key", INSTANCE_ID_COLUMN)),
                id_namespace=str(entry.get("id_namespace", branch)),
                coordinate_variant_of=str(branch),
            )
    validate_merxen_schema(sdata_obj, deep=False)
    return True


def convert_to_latest_zarr(
    raw_path: Path,
    latest_path: Path,
    *,
    quality_column_alias: str | None = None,
    platform: str | None = None,
) -> Path:
    """Migrate a raw ProSeg zarr output to SpatialData V2 format.

    Reads the raw zarr, normalizes point-table schemas to avoid pyarrow
    partition mismatches, and writes a clean V2 zarr.

    Args:
        raw_path: Path to the raw zarr (from ProSeg output).
        latest_path: Destination path for the converted zarr.
        quality_column_alias: Optional platform-native name under which to retain
            ProSeg's output ``qv`` values (for example, ``transcript_score``).
        platform: Optional source platform recorded in MerXen schema metadata.

    Returns:
        The latest_path where the converted zarr was written.
    """
    raw_path = Path(raw_path)
    latest_path = Path(latest_path)

    log_status(f"Converting to latest SpatialData layout: {raw_path} -> {latest_path}")
    if latest_path.exists():
        shutil.rmtree(latest_path)

    sdata = sd.read_zarr(raw_path)
    if all(hasattr(sdata, attribute) for attribute in ("attrs", "shapes", "tables")):
        upgrade_spatialdata_contract_in_memory(
            sdata,
            quality_column_alias=quality_column_alias,
            platform=platform,
        )
    else:
        for points_key in list(sdata.points):
            points = normalize_points_for_latest_write(
                sdata.points[points_key],
                points_key=points_key,
            )
            if quality_column_alias is not None and "qv" in points.columns:
                points[quality_column_alias] = points["qv"]
            sdata.points[points_key] = points

    write_spatialdata_zarr(sdata, latest_path)

    del sdata
    force_release(note="after latest SpatialData write")
    return latest_path


def upgrade_spatialdata_contract(
    zarr_path: Path | str,
    *,
    quality_column_alias: str | None = None,
    platform: str | None = None,
) -> bool:
    """Atomically upgrade an existing store to the current MerXen contract."""
    requested_path = Path(zarr_path)
    path = requested_path.resolve() if requested_path.is_symlink() else requested_path
    sdata_obj = sd.read_zarr(path)
    changed = upgrade_spatialdata_contract_in_memory(
        sdata_obj,
        quality_column_alias=quality_column_alias,
        platform=platform,
    )
    if not changed:
        return False

    token = uuid.uuid4().hex
    temp_path = path.parent / f".{path.name}.merxen-upgrade-{token}"
    backup_path = path.parent / f".{path.name}.merxen-backup-{token}"
    try:
        write_spatialdata_zarr(sdata_obj, temp_path)
        _copy_non_spatialdata_sidecars(path, temp_path)
        reloaded = sd.read_zarr(temp_path)
        validate_merxen_schema(reloaded, deep=False)
        os.replace(path, backup_path)
        try:
            os.replace(temp_path, path)
        except Exception:
            os.replace(backup_path, path)
            raise
        remove_path(backup_path)
    except Exception:
        if temp_path.exists() or temp_path.is_symlink():
            remove_path(temp_path)
        raise
    finally:
        del sdata_obj
        force_release(note="after SpatialData contract upgrade")
    return True


def _copy_non_spatialdata_sidecars(source: Path, destination: Path) -> None:
    """Preserve transform/spec files and other non-element payloads on rewrite."""
    element_roots = {"images", "labels", "points", "shapes", "tables"}
    metadata_names = {"zarr.json", ".zattrs", ".zgroup", ".zmetadata"}
    for child in source.iterdir():
        if child.name in element_roots or child.name in metadata_names:
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
