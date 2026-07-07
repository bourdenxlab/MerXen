"""Zarr read/write and SpatialData V2 format conversion."""

from __future__ import annotations

import os

os.environ["MPLCONFIGDIR"] = "./tmp/mpl"
os.environ["NUMBA_CACHE_DIR"] = "./tmp/numba"

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import spatialdata as sd

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
                "falling back to delete-then-write for this element only.",
                write_overwrite,
                element_key,
            )
            _delete_element_from_disk_or_path(sdata_obj, element_key, element_type)
            write_element(element_key, overwrite=False)
    else:
        path = getattr(sdata_obj, "path", None)
        write = getattr(sdata_obj, "write", None)
        if path is not None and callable(write):
            write_spatialdata_zarr(sdata_obj, Path(path), overwrite=True)

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
    int_casts: dict[str, str] = {
        "transcript_id": "uint64",
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

    write_spatialdata_zarr(sdata, latest_path)

    del sdata
    force_release(note="after latest SpatialData write")
    return latest_path
