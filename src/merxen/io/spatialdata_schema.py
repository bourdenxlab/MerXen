"""Versioned MerXen conventions layered on top of the SpatialData data model."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
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
    cell_key: str | None = None,
) -> None:
    """Register an explicit points/assignment/shape/table segmentation pairing.

    Args:
        sdata_obj: SpatialData-like object whose ``merxen_schema`` is updated.
        branch: Registry name of the segmentation branch.
        points_key: Points element the branch's assignment column lives in.
        assignment_column: Transcript column holding the branch's instance ids,
            or ``None`` when transcripts carry no assignment for it.
        shape_key: Shapes element of the branch.
        table_key: Expression table of the branch, or ``None``.
        instance_key: Column holding the instance ids of the shapes.
        id_namespace: Namespace of the branch's instance ids.
        background_column: Transcript column flagging background, if any.
        assignment_source_column: Column the assignment was derived from.
        coordinate_variant_of: Branch this one is an aligned copy of.
        legacy_aliases: Older names of the branch.
        cell_key: Shape column naming the owning cell when the branch's ids are
            not cell ids (vendor nuclei keyed by nucleus row); stored only when
            set, so registries without it stay byte-identical.
    """
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
    if cell_key is not None:
        entry["cell_key"] = str(cell_key)
    # Refreshing assignments must retain the model that produced the geometry.
    previous = schema["segmentations"].get(str(branch), {})
    source = schema["segmentations"].get(coordinate_variant_of, {})
    model_identity = previous.get(
        "model_identity",
        source.get(
            "model_identity", schema.get("model_identities", {}).get(str(branch))
        ),
    )
    if model_identity is not None:
        entry["model_identity"] = deepcopy(model_identity)
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


def canonical_shape_instance_series(
    shape_obj: Any,
    *,
    field_name: str = "shape",
) -> pd.Series:
    """Resolve positive instance IDs for canonical and legacy shape elements.

    Canonical MerXen shapes carry an explicit ``instance_id`` column. Older
    enriched stores can instead carry a zero-based ProSeg ``cell`` column or a
    prefixed Cellpose source ID while retaining an unrelated RangeIndex. This
    helper normalizes those legacy representations with the same deterministic
    rules used by enrichment, while preserving the input row index so callers
    can safely select labels after spatial subsetting.
    """
    row_index = shape_obj.index
    columns = set(getattr(shape_obj, "columns", ()))
    if INSTANCE_ID_COLUMN in columns:
        canonical = canonical_instance_series(
            shape_obj[INSTANCE_ID_COLUMN],
            field_name=f"{field_name}.{INSTANCE_ID_COLUMN}",
        )
        result = pd.Series(
            canonical.to_numpy(dtype=np.uint64),
            index=row_index,
            dtype="uint64",
        )
    else:
        candidate = next(
            (
                column
                for column in (
                    "cellpose_label",
                    SOURCE_CELL_ID_COLUMN,
                    "cell_id",
                    PROSEG_INTERNAL_ID_COLUMN,
                    "cell",
                    "cells",
                    "cell_ID",
                    "EntityID",
                    "label_id",
                    "region",
                )
                if column in columns
            ),
            None,
        )
        source = (
            pd.Series(shape_obj[candidate].to_numpy(), index=row_index)
            if candidate is not None
            else pd.Series(np.asarray(row_index), index=row_index)
        )
        if bool(source.duplicated().any()):
            raise SpatialDataContractError(
                f"{field_name} contains duplicate source identifiers"
            )

        numeric = pd.to_numeric(source, errors="coerce")
        integer_numeric = (
            len(numeric) > 0
            and numeric.notna().all()
            and (numeric >= 0).all()
            and np.array_equal(numeric.to_numpy(), np.floor(numeric.to_numpy()))
        )
        if integer_numeric:
            if bool((numeric == 0).any()):
                if candidate is None:
                    raise SpatialDataContractError(
                        f"{field_name} instance ID 0 is reserved for raster background"
                    )
                numeric = numeric + 1
            result = numeric.astype("uint64")
        else:
            text = source.astype(str).str.strip()
            prefixed = text.str.extract(
                r"^(?:cellpose|nucleus)_([0-9]+)$",
                flags=re.IGNORECASE,
                expand=False,
            )
            parsed_prefixed = pd.to_numeric(prefixed, errors="coerce")
            if parsed_prefixed.notna().all() and bool((parsed_prefixed > 0).all()):
                result = parsed_prefixed.astype("uint64")
                result.index = row_index
            else:
                ordered = {
                    source_id: index + 1
                    for index, source_id in enumerate(sorted(text.unique().tolist()))
                }
                result = text.map(ordered).astype("uint64")
                result.index = row_index

    if bool(result.duplicated().any()):
        raise SpatialDataContractError(
            f"{field_name} resolves to duplicate instance identifiers"
        )
    return result


def with_stable_transcript_ids(
    points_obj: Any,
    *,
    preserve_existing: bool = False,
    partition_lengths: Sequence[int] | None = None,
) -> Any:
    """Assign deterministic, positive, partition-order transcript identifiers.

    The IDs describe rows in the persisted MerXen points element. They are
    regenerated when ingesting or converting a source object so missing or
    silently duplicated source identifiers cannot corrupt later joins.

    Args:
        points_obj: A Dask or pandas DataFrame of transcript points.
        preserve_existing: Keep an existing ``transcript_id`` column as
            ``source_transcript_id``.
        partition_lengths: Row count of every Dask partition, in partition
            order, when the caller already knows them (for example from the
            reader's metadata). The IDs are then assigned without the
            counting pass over every partition. Each partition checks its own
            length when it is computed, so a wrong count raises instead of
            producing overlapping IDs. For a pandas DataFrame it must be
            ``[len(points_obj)]``.

    The IDs of a partition come from its Dask partition number. A selection
    such as ``.partitions[k]``, ``tail()`` or ``loc`` is pushed below this
    step and renumbers the selected partitions from 0, so it restarts the IDs
    at 1 (and, with ``partition_lengths``, fails the length check). Write or
    compute the whole frame, or read single partitions with ``to_delayed()``.

    Returns:
        The frame with a ``uint64`` ``transcript_id`` column numbered from 1.

    Raises:
        ValueError: If ``partition_lengths`` does not match the partitions.
    """
    if hasattr(points_obj, "npartitions") and hasattr(points_obj, "map_partitions"):
        if partition_lengths is None:
            lengths = (
                points_obj.map_partitions(len)
                .compute()
                .astype(np.int64, copy=False)
                .tolist()
            )
        else:
            lengths = _checked_partition_lengths(
                partition_lengths, int(points_obj.npartitions)
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
            if partition_lengths is not None and len(partition) != lengths[number]:
                raise ValueError(
                    f"Partition {number} has {len(partition)} rows; "
                    f"partition_lengths gave {lengths[number]}"
                )
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

    if partition_lengths is not None and _checked_partition_lengths(
        partition_lengths, 1
    ) != [len(points_obj)]:
        raise ValueError(
            f"partition_lengths {list(partition_lengths)} does not match "
            f"{len(points_obj)} rows"
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


def _checked_partition_lengths(
    partition_lengths: Sequence[int],
    n_partitions: int,
) -> list[int]:
    """Return caller-supplied partition lengths after checking their shape."""
    lengths = [int(length) for length in partition_lengths]
    if len(lengths) != n_partitions:
        raise ValueError(
            f"partition_lengths has {len(lengths)} entries for "
            f"{n_partitions} partitions"
        )
    if any(length < 0 for length in lengths):
        raise ValueError("partition_lengths must not be negative")
    return lengths


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

    registry = dict(attrs.get("segmentations", {}))
    for branch, entry in registry.items():
        points_key = entry.get("points")
        assignment_col = entry.get("assignment_column")
        shape_key = entry.get("shape")
        table_key = entry.get("table")
        instance_key = entry.get("instance_key", INSTANCE_ID_COLUMN)
        coordinate_variant_of = entry.get("coordinate_variant_of")

        if coordinate_variant_of is not None:
            if coordinate_variant_of not in registry:
                raise SpatialDataContractError(
                    f"{branch}: coordinate parent {coordinate_variant_of!r} "
                    "does not exist"
                )
            parent = registry[coordinate_variant_of]
            if parent.get("coordinate_variant_of") is not None:
                raise SpatialDataContractError(
                    f"{branch}: coordinate variants cannot inherit from variants"
                )
            if points_key == parent.get("points") or shape_key == parent.get("shape"):
                raise SpatialDataContractError(
                    f"{branch}: coordinate variant must use distinct points and shapes"
                )
            for semantic_key in (
                "assignment_column",
                "background_column",
                "assignment_source_column",
                "instance_key",
                "id_namespace",
            ):
                if entry.get(semantic_key) != parent.get(semantic_key):
                    raise SpatialDataContractError(
                        f"{branch}: {semantic_key} differs from coordinate parent"
                    )

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

            if coordinate_variant_of is not None:
                parent_table_key = registry[coordinate_variant_of].get("table")
                if (
                    parent_table_key is not None
                    and parent_table_key in sdata_obj.tables
                ):
                    parent_table = sdata_obj.tables[parent_table_key]
                    if table.shape != parent_table.shape:
                        raise SpatialDataContractError(
                            f"{branch}: table shape differs from coordinate parent"
                        )
                    if deep and not _matrix_values_equal(table.X, parent_table.X):
                        raise SpatialDataContractError(
                            f"{branch}: count matrix differs from coordinate parent"
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


@dataclass(frozen=True)
class StreamingPointsSummary:
    """What ``validate_merxen_schema_streaming`` counted in one points element.

    Attributes:
        points_key: Name of the points element.
        partition_rows: Row count of every partition, in partition order.
        group_rows: Row count per non-null value of the group column (empty
            when no group column was counted).
        null_group_rows: Rows whose group value is null (0 when no group
            column was counted).
        transcript_id_min: Smallest ``transcript_id`` (None when empty).
        transcript_id_max: Largest ``transcript_id`` (None when empty).
        reread_partitions: Partition reads made after the first pass, to
            check ``transcript_id`` uniqueness across partitions whose ID
            ranges overlap (a partition read for each of several ID buckets
            counts once per bucket).
    """

    points_key: str
    partition_rows: tuple[int, ...]
    group_rows: Mapping[str, int]
    null_group_rows: int
    transcript_id_min: int | None
    transcript_id_max: int | None
    reread_partitions: int

    @property
    def n_rows(self) -> int:
        """Total number of rows."""
        return sum(self.partition_rows)


@dataclass(frozen=True)
class _AssignmentCheck:
    """One registered assignment column and the shape IDs it may reference."""

    branch: str
    column: str
    shape_ids: np.ndarray


DEFAULT_MAX_REREAD_ROWS = 1 << 27
"""Default ID budget of one re-read bucket (about 2 GB at 16 B per row)."""

_ID_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def validate_merxen_schema_streaming(
    sdata_obj: Any,
    *,
    expected_partition_rows: Mapping[str, Sequence[int]] | None = None,
    group_column: str | None = None,
    expected_group_rows: Mapping[str, Mapping[str, int]] | None = None,
    max_reread_rows: int = DEFAULT_MAX_REREAD_ROWS,
) -> dict[str, StreamingPointsSummary]:
    """Run the deep schema checks one points partition at a time.

    It checks what ``validate_merxen_schema(deep=True)`` checks without
    materialising a whole points column, plus optional row-count
    reconciliation. ``validate_merxen_schema(deep=True)`` itself is
    unchanged. The checked points elements are the primary points and every
    element a segmentation branch registers (or that an expectation names):

    - everything ``validate_merxen_schema(deep=False)`` checks, including
      table IDs == shape IDs for every registered branch, and equal count
      matrices for coordinate variants;
    - ``transcript_id`` is positive and unique over the whole element (which
      implies unique per tile): unique within each partition, and partitions
      whose ID ranges overlap are re-read and checked together;
    - every non-null value of each registered assignment column is a shape ID
      of its branch (checked per partition);
    - with ``expected_partition_rows``, the row count of every partition;
    - with ``expected_group_rows``, the row count per value of
      ``group_column`` (for example per source tile, against the vendor's
      per-tile counts after the QV filter). Groups may span partitions, and
      rows with a null group value are an error.

    Partitions are read through ``to_delayed()``, so each is computed with
    its own partition number (``.partitions[k]`` renumbers the selection
    from 0, which breaks partition-numbered steps such as
    ``with_stable_transcript_ids``).

    Memory bound, for the first pass: one partition's checked columns
    (``transcript_id``, the assignment columns, the group column) and a few
    working copies of its IDs (about 4 x 8 B per row); the sorted shape IDs
    (8 B per shape); and a few integers per partition. Partitions whose ID
    ranges overlap form a cluster that is re-read. A cluster of R rows is
    split by a hash of the ID into ``ceil(R / max_reread_rows)`` buckets,
    checked one bucket per re-read of the cluster, so a bucket holds about
    ``max_reread_rows`` IDs at 16 B per row (the collected IDs and their
    concatenation) plus one partition being read. A bucket that grows past
    twice the budget raises instead of growing further. With partition-
    ordered IDs (the default contract, or vendor ordinals read in row
    ranges) there is no cluster and nothing is read twice. In the worst case
    (IDs shuffled over the whole element) every partition is read
    ``1 + ceil(n_rows / max_reread_rows)`` times.

    Args:
        sdata_obj: SpatialData object, in memory or read back from disk.
        expected_partition_rows: Expected row count per partition, keyed by
            points element. Partition numbers are those of the Dask frame as
            given; ``dd.read_parquet`` may split or merge files, so reconcile
            a store read back from disk with ``expected_group_rows`` instead.
        group_column: Points column whose values group the rows, e.g.
            ``source_tile``. Needed with ``expected_group_rows``; without
            expectations its counts are still reported when it exists.
        expected_group_rows: Expected row count per group value, keyed by
            points element. A group absent from the data counts as 0 rows,
            and a group absent from the expectations is expected to have 0.
        max_reread_rows: ID budget of one re-read bucket.

    Returns:
        A summary per points element that was read.

    Raises:
        SpatialDataContractError: On any contract violation or count mismatch.
        ValueError: If the expectations name an unknown element,
            ``expected_group_rows`` is given without ``group_column``,
            ``max_reread_rows`` is not positive, or a re-read bucket exceeds
            twice ``max_reread_rows`` without holding a duplicate.
    """
    validate_merxen_schema(sdata_obj, deep=False)
    expected_partition_rows = dict(expected_partition_rows or {})
    expected_group_rows = dict(expected_group_rows or {})
    if expected_group_rows and group_column is None:
        raise ValueError("expected_group_rows needs a group_column")
    if max_reread_rows < 1:
        raise ValueError("max_reread_rows must be positive")
    for points_key in (*expected_partition_rows, *expected_group_rows):
        if points_key not in sdata_obj.points:
            raise ValueError(f"No points element {points_key!r} to reconcile")

    attrs = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    checks: dict[str, list[_AssignmentCheck]] = {}
    primary = attrs.get("primary_points")
    if primary is not None:
        checks[str(primary)] = []
    registry = dict(attrs.get("segmentations", {}))
    for branch, entry in registry.items():
        points_key = str(entry["points"])
        branch_checks = checks.setdefault(points_key, [])
        _check_coordinate_variant_matrix(sdata_obj, registry, str(branch), entry)
        assignment_col = entry.get("assignment_column")
        if assignment_col is None:
            continue
        instance_key = entry.get("instance_key", INSTANCE_ID_COLUMN)
        shape_ids = canonical_instance_series(
            sdata_obj.shapes[entry["shape"]][instance_key],
            field_name=f"{branch}.shape.{instance_key}",
        ).to_numpy(dtype=np.uint64)
        branch_checks.append(
            _AssignmentCheck(str(branch), str(assignment_col), np.sort(shape_ids))
        )
    for points_key in (*expected_partition_rows, *expected_group_rows):
        checks.setdefault(points_key, [])

    return {
        points_key: _stream_points_checks(
            sdata_obj.points[points_key],
            points_key=points_key,
            assignment_checks=assignment_checks,
            group_column=group_column,
            expected_partition_rows=expected_partition_rows.get(points_key),
            expected_group_rows=expected_group_rows.get(points_key),
            max_reread_rows=max_reread_rows,
        )
        for points_key, assignment_checks in checks.items()
    }


def _check_coordinate_variant_matrix(
    sdata_obj: Any,
    registry: Mapping[str, Any],
    branch: str,
    entry: Mapping[str, Any],
) -> None:
    """Require a coordinate variant's count matrix to equal its parent's.

    Args:
        sdata_obj: The SpatialData object.
        registry: The segmentation registry.
        branch: Name of the branch being checked.
        entry: Its registry entry.

    Raises:
        SpatialDataContractError: If the matrices differ.
    """
    parent_key = entry.get("coordinate_variant_of")
    table_key = entry.get("table")
    if parent_key is None or table_key is None:
        return
    parent_table_key = registry[parent_key].get("table")
    if parent_table_key is None or parent_table_key not in sdata_obj.tables:
        return
    table = sdata_obj.tables[table_key]
    parent_table = sdata_obj.tables[parent_table_key]
    if not _matrix_values_equal(table.X, parent_table.X):
        raise SpatialDataContractError(
            f"{branch}: count matrix differs from coordinate parent"
        )


def _partition_readers(points: Any) -> list[Callable[[], Any]]:
    """Return one callable per partition that computes that partition.

    Dask partitions are read through ``to_delayed()``: each delayed object is
    computed in the context of the whole graph, so every task sees its true
    partition number. A pandas frame is one partition.

    Args:
        points: A Dask or pandas DataFrame or Series.

    Returns:
        The readers, in partition order.
    """
    if hasattr(points, "to_delayed"):
        return [part.compute for part in points.to_delayed()]
    return [lambda: points]


def _stream_points_checks(
    points: Any,
    *,
    points_key: str,
    assignment_checks: Sequence[_AssignmentCheck],
    group_column: str | None,
    expected_partition_rows: Sequence[int] | None,
    expected_group_rows: Mapping[str, int] | None,
    max_reread_rows: int,
) -> StreamingPointsSummary:
    """Check one points element partition by partition.

    Args:
        points: The points element.
        points_key: Its name, for messages.
        assignment_checks: Assignment columns to check against shape IDs.
        group_column: Column to count rows by, if any.
        expected_partition_rows: Expected rows per partition, if any.
        expected_group_rows: Expected rows per group value, if any.
        max_reread_rows: ID budget of one re-read bucket.

    Returns:
        What was counted.

    Raises:
        SpatialDataContractError: On a violation or count mismatch.
    """
    if TRANSCRIPT_ID_COLUMN not in points.columns:
        raise SpatialDataContractError(
            f"{points_key}: points element lacks {TRANSCRIPT_ID_COLUMN!r}"
        )
    count_groups = group_column is not None and (
        expected_group_rows is not None or group_column in points.columns
    )
    if count_groups and group_column not in points.columns:
        raise SpatialDataContractError(
            f"{points_key}: group column {group_column!r} does not exist"
        )
    columns = [TRANSCRIPT_ID_COLUMN]
    columns += [check.column for check in assignment_checks]
    if count_groups and group_column is not None:
        columns.append(group_column)
    selected = points[list(dict.fromkeys(columns))]

    partition_rows: list[int] = []
    id_ranges: list[tuple[int, int, int]] = []
    group_rows: dict[str, int] = {}
    null_group_rows = 0
    for number, read in enumerate(_partition_readers(selected)):
        partition = read()
        partition_rows.append(len(partition))
        ids = _unique_transcript_ids(
            canonical_instance_series(
                partition[TRANSCRIPT_ID_COLUMN],
                field_name=f"{points_key} partition {number}.{TRANSCRIPT_ID_COLUMN}",
            ).to_numpy(dtype=np.uint64, copy=True),
            label=f"{points_key} partition {number}",
        )
        if len(ids):
            id_ranges.append((int(ids[0]), int(ids[-1]), number))
        for check in assignment_checks:
            _check_partition_assignments(partition[check.column], check, number)
        if count_groups and group_column is not None:
            groups = partition[group_column]
            null_group_rows += int(groups.isna().sum())
            for group, count in groups.value_counts(dropna=True).items():
                if count:
                    group_rows[str(group)] = group_rows.get(str(group), 0) + int(count)
        del partition, ids

    if expected_partition_rows is not None:
        expected = [int(rows) for rows in expected_partition_rows]
        if partition_rows != expected:
            raise SpatialDataContractError(
                f"{points_key}: partition row counts {partition_rows} differ "
                f"from the expected {expected}"
            )
    if expected_group_rows is not None:
        if null_group_rows:
            raise SpatialDataContractError(
                f"{points_key}: {null_group_rows} rows have no {group_column}"
            )
        mismatched = {
            group: (group_rows.get(group, 0), int(expected_group_rows.get(group, 0)))
            for group in sorted({*group_rows, *map(str, expected_group_rows)})
            if group_rows.get(group, 0) != int(expected_group_rows.get(group, 0))
        }
        if mismatched:
            raise SpatialDataContractError(
                f"{points_key}: rows per {group_column} (found, expected) differ: "
                f"{mismatched}"
            )

    reread = 0
    clusters = _overlapping_id_clusters(id_ranges)
    if clusters:
        id_readers = _partition_readers(selected[TRANSCRIPT_ID_COLUMN])
        for cluster in clusters:
            reread += _check_cluster_unique_ids(
                [id_readers[number] for number in cluster],
                n_rows=sum(partition_rows[number] for number in cluster),
                max_reread_rows=max_reread_rows,
                label=f"{points_key} partitions {cluster}",
            )
    return StreamingPointsSummary(
        points_key=points_key,
        partition_rows=tuple(partition_rows),
        group_rows=group_rows,
        null_group_rows=null_group_rows,
        transcript_id_min=min((low for low, _, _ in id_ranges), default=None),
        transcript_id_max=max((high for _, high, _ in id_ranges), default=None),
        reread_partitions=reread,
    )


def _unique_transcript_ids(ids: np.ndarray, *, label: str) -> np.ndarray:
    """Sort transcript IDs in place and raise on a duplicate.

    Args:
        ids: ``uint64`` transcript IDs; sorted in place.
        label: Where the IDs come from, for the message.

    Returns:
        ``ids``, sorted.

    Raises:
        SpatialDataContractError: If an ID occurs more than once.
    """
    ids.sort()
    duplicated = ids[1:][ids[1:] == ids[:-1]]
    if len(duplicated):
        raise SpatialDataContractError(
            f"{label}: duplicate transcript identifiers, e.g. "
            f"{np.unique(duplicated)[:5].tolist()}"
        )
    return ids


def _check_cluster_unique_ids(
    readers: Sequence[Callable[[], Any]],
    *,
    n_rows: int,
    max_reread_rows: int,
    label: str,
) -> int:
    """Check that the IDs of overlapping partitions are unique together.

    The IDs are split into buckets by a multiplicative hash, so a repeated ID
    always lands in one bucket and the buckets are balanced whatever the ID
    pattern. Each bucket is collected over one read of every partition.

    Args:
        readers: One reader of the ``transcript_id`` column per partition.
        n_rows: Total rows of the partitions.
        max_reread_rows: ID budget of one bucket.
        label: Where the IDs come from, for messages.

    Returns:
        The number of partition reads made.

    Raises:
        SpatialDataContractError: If an ID occurs more than once.
        ValueError: If a bucket exceeds twice the budget without a duplicate.
    """
    n_buckets = max(1, -(-n_rows // max_reread_rows))
    reads = 0
    for bucket in range(n_buckets):
        chunks: list[np.ndarray] = []
        held = 0
        for read in readers:
            ids = np.asarray(read(), dtype=np.uint64)
            reads += 1
            if n_buckets > 1:
                ids = ids[_id_buckets(ids, n_buckets) == bucket]
            chunks.append(ids)
            held += len(ids)
            if held > 2 * max_reread_rows:
                _unique_transcript_ids(np.concatenate(chunks), label=label)
                raise ValueError(
                    f"{label}: an ID bucket holds more than {2 * max_reread_rows} "
                    "rows; raise max_reread_rows"
                )
        bucket_ids = np.concatenate(chunks) if chunks else np.empty(0, np.uint64)
        del chunks
        suffix = f" (bucket {bucket + 1} of {n_buckets})" if n_buckets > 1 else ""
        _unique_transcript_ids(bucket_ids, label=f"{label}{suffix}")
        del bucket_ids
    return reads


def _id_buckets(ids: np.ndarray, n_buckets: int) -> np.ndarray:
    """Return the bucket of every ID (Fibonacci hashing, wrapping uint64).

    Args:
        ids: ``uint64`` IDs.
        n_buckets: Number of buckets.

    Returns:
        The bucket number of every ID.
    """
    mixed = (ids * _ID_HASH_MULTIPLIER) >> np.uint64(32)
    return mixed % np.uint64(n_buckets)


def _check_partition_assignments(
    values: pd.Series,
    check: _AssignmentCheck,
    number: int,
) -> None:
    """Require one partition's assignments to reference existing shapes.

    Args:
        values: The partition's assignment column.
        check: The column, its branch and the branch's sorted shape IDs.
        number: Partition number, for the message.

    Raises:
        SpatialDataContractError: If an assignment is not a shape ID.
    """
    assigned = canonical_instance_series(
        pd.Series(pd.unique(values.dropna())),
        field_name=f"{check.branch}.points.{check.column}",
    ).to_numpy(dtype=np.uint64)
    missing = assigned[~np.isin(assigned, check.shape_ids)]
    if len(missing):
        raise SpatialDataContractError(
            f"{check.branch}: point assignments in partition {number} reference "
            f"missing shapes, e.g. {np.sort(missing)[:5].tolist()}"
        )


def _overlapping_id_clusters(
    id_ranges: Sequence[tuple[int, int, int]],
) -> list[list[int]]:
    """Group partitions whose ``[min, max]`` ID ranges overlap.

    Args:
        id_ranges: ``(min, max, partition number)`` per non-empty partition.

    Returns:
        The partition numbers of every cluster with more than one partition.
    """
    clusters: list[list[int]] = []
    current: list[int] = []
    current_max = -1
    for low, high, number in sorted(id_ranges):
        if current and low <= current_max:
            current.append(number)
            current_max = max(current_max, high)
            continue
        if len(current) > 1:
            clusters.append(sorted(current))
        current, current_max = [number], high
    if len(current) > 1:
        clusters.append(sorted(current))
    return clusters


def _matrix_values_equal(left: Any, right: Any) -> bool:
    """Return whether dense or sparse matrices contain exactly equal values."""
    if left.shape != right.shape:
        return False
    difference = left != right
    if hasattr(difference, "nnz"):
        return int(difference.nnz) == 0
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def is_canonical_unsigned_integer(values: Any) -> bool:
    """Return whether a Series uses a non-null unsigned integer dtype."""
    dtype = getattr(values, "dtype", None)
    return bool(
        dtype is not None
        and is_integer_dtype(dtype)
        and is_unsigned_integer_dtype(dtype)
    )
