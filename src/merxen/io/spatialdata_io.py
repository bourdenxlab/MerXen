"""Zarr read/write and SpatialData V2 format conversion."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import spatialdata as sd

from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)


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

    # Gene columns: force to plain string to avoid categorical overflow
    for gene_col in ("gene", "feature_name", "target"):
        if gene_col in cols:
            try:
                df[gene_col] = df[gene_col].astype("string")
            except Exception:  # noqa: BLE001
                df[gene_col] = df[gene_col].astype(str)

    # Integer identifier columns: prefer unsigned ints, fall back to string
    int_casts: dict[str, str] = {
        "transcript_id": "uint64",
        "assignment": "uint32",
        "cell": "uint32",
    }
    for c, target_dtype in int_casts.items():
        if c in cols:
            try:
                df[c] = df[c].astype("float64").fillna(0).astype(target_dtype)
            except Exception:  # noqa: BLE001
                try:
                    df[c] = df[c].fillna(0).astype(target_dtype)
                except Exception:  # noqa: BLE001
                    df[c] = df[c].astype("string")

    # cell_id as string for downstream assignment logic compatibility
    if "cell_id" in cols:
        try:
            df["cell_id"] = df["cell_id"].astype("string")
        except Exception:  # noqa: BLE001
            df["cell_id"] = df["cell_id"].astype(str)

    log_status(f"Normalized points schema for '{points_key}' (columns={len(cols)})")
    return df


def convert_to_latest_zarr(raw_path: Path, latest_path: Path) -> Path:
    """Migrate a raw ProSeg zarr output to SpatialData V2 format.

    Reads the raw zarr, normalizes point-table schemas to avoid pyarrow
    partition mismatches, and writes a clean V2 zarr.

    Args:
        raw_path: Path to the raw zarr (from ProSeg output).
        latest_path: Destination path for the converted zarr.

    Returns:
        The latest_path where the converted zarr was written.
    """
    raw_path = Path(raw_path)
    latest_path = Path(latest_path)

    log_status(f"Converting to latest SpatialData layout: {raw_path} -> {latest_path}")
    if latest_path.exists():
        shutil.rmtree(latest_path)

    sdata = sd.read_zarr(raw_path)

    for points_key in list(sdata.points.keys()):
        try:
            sdata.points[points_key] = normalize_points_for_latest_write(
                sdata.points[points_key], points_key=points_key
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not normalize points '%s' before latest write: %s",
                points_key,
                e,
            )

    sdata.write(latest_path)

    del sdata
    force_release(note="after latest SpatialData write")
    return latest_path
