"""Validation helpers for downstream segmentation-analysis layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import spatialdata as sd

from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    MERXEN_SCHEMA_ATTR,
    SpatialDataContractError,
    canonical_instance_series,
)

HYBRID_SEGMENTATION = "proseg_hybrid"
HYBRID_ASSIGNMENT_COLUMNS = (
    "hybrid_assignment",
    "hybrid_background",
    "hybrid_assignment_source",
)


def validate_analysis_layer(
    zarr_path: Path | str,
    *,
    platform: str,
    segmentation: str,
    table_key: str,
    shape_key: str,
) -> dict[str, Any]:
    """Validate one selected table/shape branch before downstream fan-out."""
    path = Path(zarr_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing SpatialData zarr: {path}")

    sdata_obj = sd.read_zarr(path)
    try:
        if table_key not in sdata_obj.tables:
            raise SpatialDataContractError(
                f"{segmentation}: table element {table_key!r} does not exist; "
                f"available={sorted(map(str, sdata_obj.tables.keys()))}"
            )
        if shape_key not in sdata_obj.shapes:
            raise SpatialDataContractError(
                f"{segmentation}: shape element {shape_key!r} does not exist; "
                f"available={sorted(map(str, sdata_obj.shapes.keys()))}"
            )

        table = sdata_obj.tables[table_key]
        shapes = sdata_obj.shapes[shape_key]
        attrs = dict(table.uns.get("spatialdata_attrs", {}))
        instance_key = str(attrs.get("instance_key", INSTANCE_ID_COLUMN))
        if instance_key not in table.obs.columns:
            raise SpatialDataContractError(
                f"{segmentation}: table {table_key!r} lacks instance key "
                f"{instance_key!r}"
            )
        if INSTANCE_ID_COLUMN not in shapes.columns:
            raise SpatialDataContractError(
                f"{segmentation}: shape {shape_key!r} lacks {INSTANCE_ID_COLUMN!r}"
            )

        table_ids = set(
            map(
                int,
                canonical_instance_series(
                    table.obs[instance_key],
                    field_name=f"{segmentation}.table.{instance_key}",
                ),
            )
        )
        shape_ids = set(
            map(
                int,
                canonical_instance_series(
                    shapes[INSTANCE_ID_COLUMN],
                    field_name=f"{segmentation}.shape.{INSTANCE_ID_COLUMN}",
                ),
            )
        )
        if table_ids != shape_ids:
            raise SpatialDataContractError(
                f"{segmentation}: table/shape identifier sets differ "
                f"(table={len(table_ids):,}, shape={len(shape_ids):,})"
            )

        if str(segmentation) == HYBRID_SEGMENTATION:
            _validate_hybrid_registration(
                sdata_obj,
                table_key=table_key,
                shape_key=shape_key,
            )

        return {
            "platform": str(platform).upper(),
            "segmentation": str(segmentation),
            "table_key": str(table_key),
            "shape_key": str(shape_key),
            "n_cells": len(shape_ids),
        }
    finally:
        del sdata_obj


def _validate_hybrid_registration(
    sdata_obj: Any,
    *,
    table_key: str,
    shape_key: str,
) -> None:
    schema = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    entry = dict(schema.get("segmentations", {}).get(HYBRID_SEGMENTATION, {}))
    if not entry:
        raise SpatialDataContractError(
            "proseg_hybrid: missing MerXen segmentation-branch registration"
        )
    if entry.get("table") != table_key or entry.get("shape") != shape_key:
        raise SpatialDataContractError(
            "proseg_hybrid: registered table/shape do not match the selected "
            f"layer (registered table={entry.get('table')!r}, "
            f"shape={entry.get('shape')!r})"
        )
    points_key = entry.get("points")
    if points_key not in sdata_obj.points:
        raise SpatialDataContractError(
            f"proseg_hybrid: points element {points_key!r} does not exist"
        )
    point_columns = set(map(str, sdata_obj.points[points_key].columns))
    missing = [
        column for column in HYBRID_ASSIGNMENT_COLUMNS if column not in point_columns
    ]
    if missing:
        raise SpatialDataContractError(
            "proseg_hybrid: points element is missing required assignment "
            f"columns {missing}"
        )
    expected = {
        "assignment_column": "hybrid_assignment",
        "background_column": "hybrid_background",
        "assignment_source_column": "hybrid_assignment_source",
    }
    mismatched = {
        field: (entry.get(field), value)
        for field, value in expected.items()
        if entry.get(field) != value
    }
    if mismatched:
        raise SpatialDataContractError(
            f"proseg_hybrid: invalid assignment registration {mismatched}"
        )
