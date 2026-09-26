"""Transcript I/O and common DataFrame utility functions.

Consolidates the duplicated helper functions from the original notebook into
single, robust implementations.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from merxen.control_features import (
    classify_control_transcripts,
    is_registered_control_name,
)
from merxen.memory import enforce_memory_limit, log_status

logger = logging.getLogger(__name__)


def to_pandas(df_like: Any) -> pd.DataFrame:
    """Convert a Dask or pandas-like object to a concrete pandas DataFrame.

    Args:
        df_like: A pandas DataFrame, Dask DataFrame, or similar.

    Returns:
        A pandas DataFrame (computed if lazy).
    """
    if hasattr(df_like, "compute") and not isinstance(df_like, pd.DataFrame):
        return df_like.compute()
    return df_like.copy()


def resolve_col(
    df_like: Any,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """Find the first existing column from a list of candidates.

    Args:
        df_like: A DataFrame-like object with a .columns attribute.
        candidates: Column names to search for, in priority order.
        required: If True, raise KeyError when no candidate is found.

    Returns:
        The first matching column name, or None if not required and none found.

    Raises:
        KeyError: If required is True and no candidate column exists.
    """
    cols = set(map(str, list(df_like.columns)))
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(
            f"Could not resolve required column from candidates={candidates}. "
            f"Available={sorted(cols)}"
        )
    return None


def first_existing_col(df_like: Any, cols: list[str]) -> str | None:
    """Find the first column that exists in a DataFrame.

    Args:
        df_like: A DataFrame-like object with a .columns attribute.
        cols: Column names to search for, in priority order.

    Returns:
        The first matching column name, or None if none found.
    """
    return resolve_col(df_like, cols, required=False)


def assignment_mask(series: pd.Series) -> pd.Series:
    """Boolean mask identifying assigned transcript values.

    Handles both numeric cell IDs and string-based assignment columns.
    Numeric columns with nulls are treated as nullable cell IDs, where any
    non-null value, including 0, is a valid assignment.

    Args:
        series: A pandas Series of cell assignment values.

    Returns:
        Boolean Series where True means the transcript is assigned to a cell.
    """
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        if bool(vals.isna().any()):
            return vals.notna()
        return vals != 0
    s = series.astype(str).str.strip().str.lower()
    return (
        ~s.isin({"0", "0.0", "", "none", "nan", "null", "unassigned"})
    ) & series.notna()


def background_mask(series: pd.Series) -> pd.Series:
    """Boolean mask identifying background/unassigned transcript values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(True).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(1)
        return vals != 0

    values = series.astype("string").str.strip().str.lower()
    true_values = values.isin({"true", "t", "1", "yes", "y"})
    false_values = values.isin({"false", "f", "0", "no", "n"})
    unknown_values = values.isna() | ~(true_values | false_values)
    return true_values | unknown_values


def assignment_mask_from_points(
    points_df: pd.DataFrame,
    *,
    assign_col: str | None = None,
    background_col: str = "background",
) -> pd.Series:
    """Boolean mask identifying assigned transcripts from a points partition."""
    cols = set(map(str, list(points_df.columns)))
    if background_col in cols:
        return ~background_mask(points_df[background_col])
    if assign_col is None:
        raise KeyError("No assignment or background column was provided.")
    return assignment_mask(points_df[assign_col])


def iter_points_chunks(
    points_obj: Any,
    columns: list[str],
    chunk_rows: int = 1_000_000,
    desc: str = "Transcript chunks",
) -> Generator[pd.DataFrame, None, None]:
    """Lazily yield chunks of transcript points from a Dask or pandas object.

    Args:
        points_obj: A Dask or pandas DataFrame of transcript points.
        columns: Columns to select from each chunk.
        chunk_rows: Number of rows per chunk (for pandas fallback).
        desc: Progress bar description.

    Yields:
        pandas DataFrames with the requested columns.
    """
    if hasattr(points_obj, "npartitions") and hasattr(points_obj, "partitions"):
        n_parts = int(points_obj.npartitions)
        pbar = tqdm(range(n_parts), total=n_parts, desc=desc, unit="part")
        for pidx in pbar:
            part = points_obj.partitions[pidx][columns].compute()
            yield part
        return

    if hasattr(points_obj, "compute") and not isinstance(points_obj, pd.DataFrame):
        log_status(
            "[Points] Materializing non-partitioned lazy points object (fallback path)"
        )
        points_obj = points_obj[columns].compute()

    n_rows = len(points_obj)
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    pbar = tqdm(range(0, n_rows, chunk_rows), total=n_chunks, desc=desc, unit="chunk")
    for start in pbar:
        stop = min(start + chunk_rows, n_rows)
        yield points_obj.iloc[start:stop][columns].copy()


def write_proseg_csv_from_points(
    points_obj: Any,
    csv_path: Path,
    masks: np.ndarray,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
    x_col: str,
    y_col: str,
    z_col: str | None,
    gene_col: str,
    *,
    qv_col: str | None = None,
    min_qv: float | None = None,
    excluded_gene_pattern: str | None = None,
    control_platform: str | None = None,
    is_gene_col: str | None = None,
    codeword_category_col: str | None = None,
    chunk_rows: int = 1_000_000,
    dataset_name: str = "DATASET",
    status_every_chunks: int = 5,
    memory_check_every_chunks: int = 5,
    max_ram_gb: float = 600.0,
    warn_ram_gb: float = 560.0,
) -> dict[str, Any]:
    """Write a ProSeg-compatible transcript CSV with cell assignments from masks.

    Lazily processes transcripts in chunks, assigns each to a Cellpose mask cell,
    and writes the result to a CSV file suitable for ProSeg input.

    Args:
        points_obj: Transcript points (Dask or pandas DataFrame).
        csv_path: Output CSV file path.
        masks: Labeled 2D mask array (cell_id > 0 = assigned).
        x_transform: Affine transform (a, b, tx) for x: pixel to micron.
        y_transform: Affine transform (a, b, ty) for y: pixel to micron.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        z_col: Column name for z coordinates (None to fill with zeros).
        gene_col: Column name for gene/feature labels.
        qv_col: Optional quality value column for filtering.
        min_qv: Minimum QV threshold (requires qv_col).
        excluded_gene_pattern: Optional regular expression matched against the
            start of feature names. Matching transcripts are omitted.
        control_platform: Platform (``"XENIUM"`` or ``"MERSCOPE"``) whose
            control features are omitted using
            :func:`merxen.control_features.classify_control_transcripts`.
            ``None`` keeps control features.
        is_gene_col: Optional Xenium ``is_gene`` column; preferred over name
            rules when ``control_platform`` is set.
        codeword_category_col: Optional Xenium ``codeword_category`` column,
            used where ``is_gene`` is unavailable.
        chunk_rows: Rows per processing chunk.
        dataset_name: Name for log messages.
        status_every_chunks: Log status every N chunks.
        memory_check_every_chunks: Check memory every N chunks.
        max_ram_gb: Hard memory limit in GB.
        warn_ram_gb: Soft memory warning threshold in GB.

    Returns:
        Dict with csv_path, n_input, n_written, n_seeded, pct_seeded,
        n_excluded_genes (all transcripts dropped by feature filters),
        n_excluded_controls, excluded_control_counts (per control feature)
        and kept_by_feature_type_counts (per feature not treated as a control
        because its feature type says gene although a control-name rule
        matches).
    """
    # Import here to avoid circular dependency
    from merxen.segmentation.cellpose import (
        assign_labels_from_masks,
        invert_mask_affine,
    )

    csv_path = Path(csv_path)
    if csv_path.exists():
        csv_path.unlink()

    a_inv, b = invert_mask_affine(x_transform, y_transform)

    cols = [x_col, y_col, gene_col]
    if z_col is not None:
        cols.append(z_col)
    if qv_col is not None and qv_col not in cols:
        cols.append(qv_col)
    if control_platform is not None:
        for feature_type_col in (is_gene_col, codeword_category_col):
            if feature_type_col is not None and feature_type_col not in cols:
                cols.append(feature_type_col)

    n_input = 0
    n_written = 0
    n_seeded = 0
    n_excluded_genes = 0
    n_excluded_controls = 0
    excluded_control_counts: Counter[str] = Counter()
    kept_by_feature_type_counts: Counter[str] = Counter()
    next_transcript_id = 0
    header_written = False

    log_status(f"[{dataset_name}] Writing ProSeg CSV lazily to: {csv_path}")
    chunk_iter = iter_points_chunks(
        points_obj,
        columns=cols,
        chunk_rows=chunk_rows,
        desc=f"[{dataset_name}] transcript chunks",
    )

    for i, chunk in enumerate(chunk_iter, start=1):
        n_input += len(chunk)

        x_vals = pd.to_numeric(chunk[x_col], errors="coerce").to_numpy(np.float64)
        y_vals = pd.to_numeric(chunk[y_col], errors="coerce").to_numpy(np.float64)

        if z_col is None:
            z_vals = np.zeros(len(chunk), dtype=np.float32)
        else:
            z_vals = (
                pd.to_numeric(chunk[z_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32)
            )

        gene_vals = chunk[gene_col].astype(str).to_numpy(dtype=object)

        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        valid &= pd.notna(gene_vals)
        valid &= gene_vals != ""

        if excluded_gene_pattern is not None:
            excluded = pd.Series(gene_vals, dtype="string").str.match(
                excluded_gene_pattern,
                na=False,
            )
            excluded_values = excluded.to_numpy(dtype=bool)
            n_excluded_genes += int(np.count_nonzero(valid & excluded_values))
            valid &= ~excluded_values

        if control_platform is not None:
            flags = classify_control_transcripts(
                gene_vals,
                platform=control_platform,
                is_gene=None if is_gene_col is None else chunk[is_gene_col],
                codeword_category=(
                    None
                    if codeword_category_col is None
                    else chunk[codeword_category_col]
                ),
            )
            dropped = valid & flags.control
            n_dropped = int(np.count_nonzero(dropped))
            _count_features(excluded_control_counts, gene_vals[dropped])
            _count_features(
                kept_by_feature_type_counts,
                gene_vals[valid & flags.kept_by_feature_type],
            )
            n_excluded_controls += n_dropped
            n_excluded_genes += n_dropped
            valid &= ~flags.control

        qv_vals: np.ndarray | None = None
        if qv_col is not None and min_qv is not None:
            qv_vals = pd.to_numeric(chunk[qv_col], errors="coerce").to_numpy(np.float64)
            valid &= np.isfinite(qv_vals) & (qv_vals >= float(min_qv))
        elif qv_col is not None:
            qv_vals = pd.to_numeric(chunk[qv_col], errors="coerce").to_numpy(np.float64)
            valid &= np.isfinite(qv_vals)

        if np.any(valid):
            xv = x_vals[valid]
            yv = y_vals[valid]
            zv = z_vals[valid]
            gv = gene_vals[valid]
            transcript_ids = np.arange(
                next_transcript_id,
                next_transcript_id + len(xv),
                dtype=np.uint64,
            )
            next_transcript_id += len(xv)

            labels = assign_labels_from_masks(xv, yv, masks, a_inv=a_inv, b=b)

            out_df = pd.DataFrame(
                {
                    "transcript_id": transcript_ids,
                    "x_micron": xv.astype(np.float32, copy=False),
                    "y_micron": yv.astype(np.float32, copy=False),
                    "z_micron": zv.astype(np.float32, copy=False),
                    "feature_name": gv.astype(str),
                    "cell_id": labels.astype(np.int32, copy=False),
                }
            )
            if qv_vals is not None:
                out_df["qv"] = qv_vals[valid].astype(np.float32, copy=False)

            out_df.to_csv(
                csv_path,
                mode="w" if not header_written else "a",
                header=not header_written,
                index=False,
            )

            header_written = True
            n_written += len(out_df)
            n_seeded += int((labels != 0).sum())

            del xv, yv, zv, gv, transcript_ids, labels, out_df

        if i % status_every_chunks == 0:
            log_status(
                f"[{dataset_name}] chunk={i} input={n_input:,} "
                f"written={n_written:,} seeded={n_seeded:,}"
            )

        if i % memory_check_every_chunks == 0:
            enforce_memory_limit(
                stage=f"{dataset_name} transcript chunk {i}",
                max_gb=max_ram_gb,
                warn_gb=warn_ram_gb,
            )

        del chunk, x_vals, y_vals, z_vals, gene_vals, qv_vals, valid

    if not header_written:
        raise RuntimeError(
            f"[{dataset_name}] No transcripts were written to {csv_path}. "
            "Check coordinate/qv filters and selected columns."
        )

    pct_seeded = 100.0 * n_seeded / max(n_written, 1)
    log_status(
        f"[{dataset_name}] ProSeg CSV complete: {n_written:,} rows written, "
        f"{n_seeded:,} seeded ({pct_seeded:.2f}%), input={n_input:,}"
    )
    if control_platform is not None:
        _log_excluded_controls(
            dataset_name,
            control_platform,
            n_excluded_controls,
            excluded_control_counts,
            kept_by_feature_type_counts,
        )

    return {
        "csv_path": csv_path,
        "n_input": int(n_input),
        "n_written": int(n_written),
        "n_seeded": int(n_seeded),
        "pct_seeded": float(pct_seeded),
        "n_excluded_genes": int(n_excluded_genes),
        "n_excluded_controls": int(n_excluded_controls),
        "excluded_control_counts": dict(excluded_control_counts),
        "kept_by_feature_type_counts": dict(kept_by_feature_type_counts),
    }


def _count_features(counter: Counter[str], names: np.ndarray) -> None:
    """Add per-feature transcript counts for ``names`` to ``counter``."""
    if names.size == 0:
        return
    unique, counts = np.unique(names.astype(str), return_counts=True)
    counter.update(dict(zip(unique.tolist(), counts.tolist(), strict=True)))


def _log_excluded_controls(
    dataset_name: str,
    platform: str,
    n_excluded: int,
    counts: Counter[str],
    kept_by_feature_type: Counter[str],
) -> None:
    """Log the control features dropped from a ProSeg CSV.

    Controls recognised only through a feature-type column are named in a
    warning: the registry in :mod:`merxen.control_features` lacks their names.
    Features not treated as controls because their feature type says gene
    although a control-name rule matches are named in a second warning.
    """
    log_status(
        f"[{dataset_name}] Excluded {n_excluded:,} control transcripts "
        f"({len(counts):,} features) from the ProSeg CSV"
    )
    unregistered = sorted(
        name for name in counts if not is_registered_control_name(name, platform)
    )
    if unregistered:
        logger.warning(
            "[%s] %d control features were recognised only by their feature "
            "type; add their names to merxen.control_features: %s",
            dataset_name,
            len(unregistered),
            ", ".join(unregistered[:20]),
        )
    if kept_by_feature_type:
        kept = sorted(kept_by_feature_type.items())
        logger.warning(
            "[%s] %d features were not treated as controls because their "
            "feature type says gene, although their names match a control-name "
            "rule (%s transcripts): %s",
            dataset_name,
            len(kept),
            f"{sum(kept_by_feature_type.values()):,}",
            ", ".join(f"{name} ({count:,})" for name, count in kept[:20]),
        )
