"""Versioned MerXen conventions layered on top of the SpatialData data model."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_unsigned_integer_dtype

MERXEN_SCHEMA_ATTR = "merxen_schema"
MERXEN_SCHEMA_VERSION = "1.0.0"

INSTANCE_ID_COLUMN = "instance_id"
SOURCE_CELL_ID_COLUMN = "source_cell_id"
PROSEG_INTERNAL_ID_COLUMN = "proseg_internal_id"
TRANSCRIPT_ID_COLUMN = "transcript_id"
SOURCE_TRANSCRIPT_ID_COLUMN = "source_transcript_id"

PROSEG_ASSIGNMENT_COLUMN = "assignment"
PROSEG_GEOMETRY_ASSIGNMENT_COLUMN = "proseg_geometry_assignment"
LEGACY_PROSEG_MASK_ASSIGNMENT_COLUMN = "proseg_mask_assignment"

PROSEG_ID_NAMESPACE = "cellpose_label_v1"
ORIGINAL_ID_NAMESPACE = "source_segmentation_v1"


class SpatialDataContractError(ValueError):
    """Raised when a MerXen SpatialData identifier contract is inconsistent."""


def package_versions() -> dict[str, str]:
    """Return the storage-relevant package versions for output provenance."""
    versions: dict[str, str] = {}
    for package in (
        "merxen",
        "spatialdata",
        "spatialdata-io",
        "zarr",
        "pyarrow",
        "dask",
        "anndata",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def choose_primary_points_key(sdata_obj: Any) -> str | None:
    """Resolve the canonical native points element without relying on dict order."""
    attrs = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    configured = attrs.get("primary_points")
    points = getattr(sdata_obj, "points", {})
    if isinstance(configured, str) and configured in points:
        return configured
    if "transcripts" in points:
        return "transcripts"
    native = [str(key) for key in points if not str(key).endswith("_aligned_nonrigid")]
    if native:
        return sorted(native)[0]
    keys = sorted(map(str, points))
    return keys[0] if keys else None


def stamp_merxen_schema(
    sdata_obj: Any,
    *,
    primary_points_key: str | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """Attach or update versioned MerXen storage metadata."""
    root_attrs = getattr(sdata_obj, "attrs", None)
    if root_attrs is None:
        root_attrs = {}
        sdata_obj.attrs = root_attrs

    existing = dict(root_attrs.get(MERXEN_SCHEMA_ATTR, {}))
    registry = dict(existing.get("segmentations", {}))
    primary = primary_points_key or choose_primary_points_key(sdata_obj)
    payload: dict[str, Any] = {
        **existing,
        "schema_version": MERXEN_SCHEMA_VERSION,
        "primary_points": primary,
        "instance_id": {
            "column": INSTANCE_ID_COLUMN,
            "dtype": "uint64",
            "minimum": 1,
            "raster_background": 0,
            "unassigned": None,
        },
        "transcript_id": {
            "column": TRANSCRIPT_ID_COLUMN,
            "dtype": "uint64",
            "minimum": 1,
        },
        "writer_versions": package_versions(),
        "segmentations": registry,
    }
    if platform is not None:
        payload["platform"] = str(platform).upper()
    root_attrs[MERXEN_SCHEMA_ATTR] = payload
    return payload


def register_segmentation_branch(
    sdata_obj: Any,
    branch: str,
    *,
    points_key: str,
    assignment_column: str | None,
    shape_key: str,
    table_key: str | None,
    instance_key: str = INSTANCE_ID_COLUMN,
    id_namespace: str,
    background_column: str | None = None,
    assignment_source_column: str | None = None,
    coordinate_variant_of: str | None = None,
    legacy_aliases: tuple[str, ...] = (),
) -> None:
    """Register an explicit points/assignment/shape/table segmentation pairing."""
    schema = stamp_merxen_schema(sdata_obj)
    if schema.get("primary_points") is None:
        schema["primary_points"] = str(points_key)
    entry: dict[str, Any] = {
        "points": str(points_key),
        "assignment_column": assignment_column,
        "shape": str(shape_key),
        "table": None if table_key is None else str(table_key),
        "instance_key": str(instance_key),
        "id_namespace": str(id_namespace),
        "background_column": background_column,
        "assignment_source_column": assignment_source_column,
        "legacy_aliases": list(legacy_aliases),
    }
    if coordinate_variant_of is not None:
        entry["coordinate_variant_of"] = str(coordinate_variant_of)
    schema["segmentations"][str(branch)] = entry
    sdata_obj.attrs[MERXEN_SCHEMA_ATTR] = schema


def canonical_instance_series(
    values: Any,
    *,
    field_name: str = INSTANCE_ID_COLUMN,
    allow_null: bool = False,
) -> pd.Series:
    """Convert values to positive canonical instance identifiers."""
    original = pd.Series(values, copy=False)
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if bool(invalid.any()):
        examples = original.loc[invalid].astype(str).head(5).tolist()
        raise SpatialDataContractError(
            f"{field_name} contains non-numeric identifiers: {examples}"
        )
    if not allow_null and bool(numeric.isna().any()):
        raise SpatialDataContractError(f"{field_name} contains missing identifiers")
    nonnull = numeric.dropna()
    if bool((nonnull <= 0).any()):
        examples = nonnull.loc[nonnull <= 0].head(5).tolist()
        raise SpatialDataContractError(
            f"{field_name} must contain positive identifiers; found {examples}"
        )
    dtype = "UInt64" if allow_null else "uint64"
    return numeric.astype(dtype)


def with_stable_transcript_ids(
    points_obj: Any,
    *,
    preserve_existing: bool = False,
) -> Any:
    """Assign deterministic, positive, partition-order transcript identifiers.

    The IDs describe rows in the persisted MerXen points element. They are
    regenerated when ingesting or converting a source object so missing or
    silently duplicated source identifiers cannot corrupt later joins.
    """
    if hasattr(points_obj, "npartitions") and hasattr(points_obj, "map_partitions"):
        lengths = (
            points_obj.map_partitions(len)
            .compute()
            .astype(np.int64, copy=False)
            .tolist()
        )
        offsets: list[int] = []
        next_id = 1
        for length in lengths:
            offsets.append(next_id)
            next_id += int(length)

        meta = points_obj._meta.copy()
        preserve_source = (
            preserve_existing
            and TRANSCRIPT_ID_COLUMN in points_obj.columns
            and SOURCE_TRANSCRIPT_ID_COLUMN not in points_obj.columns
        )
        if preserve_source:
            meta[SOURCE_TRANSCRIPT_ID_COLUMN] = meta[TRANSCRIPT_ID_COLUMN]
        meta[TRANSCRIPT_ID_COLUMN] = pd.Series(dtype="uint64")

        def _assign_partition(
            partition: pd.DataFrame,
            *,
            partition_info: dict[str, Any] | None = None,
        ) -> pd.DataFrame:
            number = 0 if partition_info is None else int(partition_info["number"])
            start = offsets[number]
            result = partition.copy()
            if preserve_source:
                result[SOURCE_TRANSCRIPT_ID_COLUMN] = result[TRANSCRIPT_ID_COLUMN]
            result[TRANSCRIPT_ID_COLUMN] = np.arange(
                start,
                start + len(result),
                dtype=np.uint64,
            )
            return result

        return points_obj.map_partitions(
            _assign_partition,
            meta=meta,
            partition_info=True,
        )

    result = points_obj.copy()
    if (
        preserve_existing
        and TRANSCRIPT_ID_COLUMN in result.columns
        and SOURCE_TRANSCRIPT_ID_COLUMN not in result.columns
    ):
        result[SOURCE_TRANSCRIPT_ID_COLUMN] = result[TRANSCRIPT_ID_COLUMN]
    result[TRANSCRIPT_ID_COLUMN] = np.arange(
        1,
        len(result) + 1,
        dtype=np.uint64,
    )
    return result


def validate_merxen_schema(
    sdata_obj: Any,
    *,
    deep: bool = False,
) -> None:
    """Validate registered assignment/shape/table referential integrity.

    Args:
        sdata_obj: SpatialData-like object to validate.
        deep: Also scan point assignments and transcript IDs. This can require
            reading every Parquet partition for large datasets.
    """
    attrs = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    if attrs.get("schema_version") != MERXEN_SCHEMA_VERSION:
        raise SpatialDataContractError(
            "SpatialData object is missing the current MerXen schema version"
        )

    primary = attrs.get("primary_points")
    if primary is not None and primary not in sdata_obj.points:
        raise SpatialDataContractError(
            f"Primary points element {primary!r} does not exist"
        )

    for branch, entry in dict(attrs.get("segmentations", {})).items():
        points_key = entry.get("points")
        assignment_col = entry.get("assignment_column")
        shape_key = entry.get("shape")
        table_key = entry.get("table")
        instance_key = entry.get("instance_key", INSTANCE_ID_COLUMN)

        if points_key not in sdata_obj.points:
            raise SpatialDataContractError(
                f"{branch}: points element {points_key!r} does not exist"
            )
        if shape_key not in sdata_obj.shapes:
            raise SpatialDataContractError(
                f"{branch}: shape element {shape_key!r} does not exist"
            )
        points = sdata_obj.points[points_key]
        if TRANSCRIPT_ID_COLUMN not in points.columns:
            raise SpatialDataContractError(
                f"{branch}: points element lacks {TRANSCRIPT_ID_COLUMN!r}"
            )
        if assignment_col is not None and assignment_col not in points.columns:
            raise SpatialDataContractError(
                f"{branch}: assignment column {assignment_col!r} does not exist"
            )

        shapes = sdata_obj.shapes[shape_key]
        if instance_key not in shapes.columns:
            raise SpatialDataContractError(f"{branch}: shape lacks {instance_key!r}")
        shape_ids = canonical_instance_series(
            shapes[instance_key],
            field_name=f"{branch}.shape.{instance_key}",
        )
        if bool(shape_ids.duplicated().any()):
            raise SpatialDataContractError(f"{branch}: duplicate shape identifiers")
        index_ids = canonical_instance_series(
            shapes.index,
            field_name=f"{branch}.shape.index",
        )
        if not np.array_equal(
            shape_ids.to_numpy(dtype=np.uint64),
            index_ids.to_numpy(dtype=np.uint64),
        ):
            raise SpatialDataContractError(
                f"{branch}: shape index and {instance_key!r} differ"
            )

        table_ids: set[int] | None = None
        if table_key is not None:
            if table_key not in sdata_obj.tables:
                raise SpatialDataContractError(
                    f"{branch}: table element {table_key!r} does not exist"
                )
            table = sdata_obj.tables[table_key]
            table_attrs = dict(table.uns.get("spatialdata_attrs", {}))
            if table_attrs.get("instance_key") != instance_key:
                raise SpatialDataContractError(
                    f"{branch}: table instance_key is not {instance_key!r}"
                )
            table_values = canonical_instance_series(
                table.obs[instance_key],
                field_name=f"{branch}.table.{instance_key}",
            )
            if bool(table_values.duplicated().any()):
                raise SpatialDataContractError(f"{branch}: duplicate table identifiers")
            table_ids = set(map(int, table_values))
            if table_ids != set(map(int, shape_ids)):
                raise SpatialDataContractError(
                    f"{branch}: shape and table identifier sets differ"
                )

        if not deep:
            continue
        transcript_ids = points[TRANSCRIPT_ID_COLUMN]
        if hasattr(transcript_ids, "compute"):
            transcript_ids = transcript_ids.compute()
        transcript_values = canonical_instance_series(
            transcript_ids,
            field_name=f"{points_key}.{TRANSCRIPT_ID_COLUMN}",
        )
        if bool(transcript_values.duplicated().any()):
            raise SpatialDataContractError(
                f"{points_key}: duplicate transcript identifiers"
            )
        if assignment_col is not None:
            assignments = points[assignment_col]
            if hasattr(assignments, "dropna"):
                assignments = assignments.dropna()
            if hasattr(assignments, "drop_duplicates"):
                assignments = assignments.drop_duplicates()
            if hasattr(assignments, "compute"):
                assignments = assignments.compute()
            assignment_ids = set(
                map(
                    int,
                    canonical_instance_series(
                        assignments,
                        field_name=f"{branch}.points.{assignment_col}",
                    ),
                )
            )
            if not assignment_ids.issubset(set(map(int, shape_ids))):
                raise SpatialDataContractError(
                    f"{branch}: point assignments reference missing shapes"
                )


def is_canonical_unsigned_integer(values: Any) -> bool:
    """Return whether a Series uses a non-null unsigned integer dtype."""
    dtype = getattr(values, "dtype", None)
    return bool(
        dtype is not None
        and is_integer_dtype(dtype)
        and is_unsigned_integer_dtype(dtype)
    )
