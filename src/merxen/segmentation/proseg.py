"""ProSeg subprocess orchestration vendored and trimmed from MOSAIK."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import zarr
from numpy.typing import DTypeLike

from merxen.path_utils import remove_path
from merxen.segmentation.tempdirs import large_temp_root

logger = logging.getLogger(__name__)

# Negative translations are lifted by whole multiples of this round number of
# microns, which is also a multiple of the usual voxel sizes (0.5 to 4 um).
_COORDINATE_SHIFT_STEP_UM = 100.0
# Linear affine terms this small relative to the pixel size are float noise
# from inverting the micron-to-pixel matrix, not a rotation or flip.
_LINEAR_TERM_NOISE_TOLERANCE = 1e-9
COORDINATE_SHIFT_SIDECAR = "proseg_coordinate_shift.json"
_POINT_COLUMNS_BY_AXIS = {"x": ("x", "observed_x"), "y": ("y", "observed_y")}
_TABLE_CENTROID_COLUMNS = {"x": "centroid_x", "y": "centroid_y"}
_TRANSCRIPT_CHUNK_ROWS = 1_000_000


@dataclass(frozen=True)
class ProsegCoordinateShift:
    """Micron offset added to transcript x/y so ProSeg sees no negative values.

    Attributes:
        x: Offset added to transcript x coordinates and the x translation.
        y: Offset added to transcript y coordinates and the y translation.
    """

    x: float = 0.0
    y: float = 0.0

    @property
    def is_identity(self) -> bool:
        """Whether no shift is needed."""
        return self.x == 0.0 and self.y == 0.0


def _has_minus_sign(value: float) -> bool:
    """Whether ``str(value)`` would start with ``-`` (including ``-0.0``)."""
    return math.copysign(1.0, float(value)) < 0.0


def proseg_safe_affine(
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], ProsegCoordinateShift
]:
    """Make a mask-to-micron affine passable on the ProSeg command line.

    ProSeg's argument parser reads any value starting with ``-`` as a flag and
    has no syntax for a negative ``--cellpose-*-transform`` value. MERSCOPE
    mosaics can start at negative microns, which gives a negative translation.
    A negative translation is lifted by a whole multiple of
    ``_COORDINATE_SHIFT_STEP_UM`` and the same offset must then be added to
    the transcript coordinates (and removed from ProSeg's outputs). Values
    that are already non-negative are returned unchanged, so the common case
    builds the same command as before.

    Args:
        x_transform: Mask-pixel-to-micron terms ``(a, b, c)`` for x.
        y_transform: Mask-pixel-to-micron terms ``(a, b, c)`` for y.

    Returns:
        The non-negative x and y transforms, and the shift they require.

    Raises:
        ValueError: If a linear term is genuinely negative (a flip or
            rotation), which no translation can make non-negative.
    """
    safe_terms: list[tuple[float, float, float]] = []
    offsets: list[float] = []
    for axis, transform in (("x", x_transform), ("y", y_transform)):
        scale = max(abs(float(transform[0])), abs(float(transform[1])))
        linear: list[float] = []
        for term in transform[:2]:
            if not _has_minus_sign(term):
                linear.append(term)
            elif abs(term) <= _LINEAR_TERM_NOISE_TOLERANCE * scale:
                # Covers -0.0 too, which ProSeg would also read as a flag.
                linear.append(0.0)
            else:
                raise ValueError(
                    f"ProSeg cannot take the negative linear term {term} in the "
                    f"{axis} mask-to-micron affine {list(transform)}; flipped or "
                    "rotated masks are not supported."
                )
        translation = transform[2]
        offset = 0.0
        if _has_minus_sign(translation):
            offset = (
                math.ceil(-translation / _COORDINATE_SHIFT_STEP_UM)
                * _COORDINATE_SHIFT_STEP_UM
            )
            # Exact float sums keep this >= 0 because offset >= -translation.
            translation = translation + offset
        safe_terms.append((linear[0], linear[1], translation))
        offsets.append(offset)
    shift = ProsegCoordinateShift(x=offsets[0], y=offsets[1])
    return safe_terms[0], safe_terms[1], shift


def _write_shifted_transcripts(
    source: Path,
    destination: Path,
    columns: list[str],
    x_col: str,
    y_col: str,
    shift: ProsegCoordinateShift,
) -> None:
    """Copy the ProSeg transcript CSV with ``shift`` added to x and y.

    Columns are read as text so identifiers such as the ``0`` unassigned cell
    id keep their exact spelling; only the shifted columns are rewritten.
    """
    chunks = pd.read_csv(
        source,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        chunksize=_TRANSCRIPT_CHUNK_ROWS,
    )
    is_first = True
    for chunk in chunks:
        for column, offset in ((x_col, shift.x), (y_col, shift.y)):
            if offset != 0.0:
                chunk[column] = pd.to_numeric(chunk[column]) + offset
        chunk.to_csv(
            destination, index=False, header=is_first, mode="w" if is_first else "a"
        )
        is_first = False


def _rewrite_parquet(parquet_path: Path, table: pa.Table) -> None:
    """Write ``table`` over ``parquet_path`` keeping its file-level metadata.

    ProSeg stores the GeoParquet ``geo`` key in the file's key-value metadata
    but not in the embedded Arrow schema, so ``pq.read_table`` drops it and
    geopandas (and so spatialdata) cannot read a naive rewrite.
    """
    metadata = dict(pq.read_metadata(parquet_path).metadata or {})
    metadata.pop(b"ARROW:schema", None)
    if b"geo" in metadata:
        geo = json.loads(metadata[b"geo"])
        for column in geo.get("columns", {}).values():
            # An optional bounding box would be stale after the shift.
            column.pop("bbox", None)
        metadata[b"geo"] = json.dumps(geo).encode()
    metadata.update(table.schema.metadata or {})
    pq.write_table(
        table.replace_schema_metadata(metadata), parquet_path, compression="zstd"
    )


def _shift_parquet_columns(parquet_path: Path, offsets: dict[str, float]) -> set[str]:
    """Subtract per-column offsets in place; return the columns changed."""
    table = pq.read_table(parquet_path)
    changed: set[str] = set()
    for column, offset in offsets.items():
        index = table.schema.get_field_index(column)
        if index < 0 or offset == 0.0:
            continue
        values = table.column(index).to_numpy()
        restored = (values.astype(np.float64) - offset).astype(values.dtype)
        table = table.set_column(index, table.schema.field(index), pa.array(restored))
        changed.add(column)
    if changed:
        _rewrite_parquet(parquet_path, table)
    return changed


def _shift_parquet_geometries(parquet_path: Path, shift: ProsegCoordinateShift) -> None:
    """Subtract ``shift`` from every vertex of a WKB geometry parquet in place."""
    table = pq.read_table(parquet_path)
    index = table.schema.get_field_index("geometry")
    if index < 0:
        raise RuntimeError(f"No geometry column in ProSeg shapes {parquet_path}")
    field = table.schema.field(index)
    geometries = shapely.from_wkb(
        np.asarray(table.column(index).to_pylist(), dtype=object)
    )
    offset = np.asarray([shift.x, shift.y], dtype=np.float64)
    restored = shapely.transform(geometries, lambda coords: coords - offset)
    wkb = pa.array(list(shapely.to_wkb(restored)), type=field.type)
    _rewrite_parquet(parquet_path, table.set_column(index, field, wkb))


def _subtract_from_zarr_array(array_path: Path, offset: float | np.ndarray) -> bool:
    """Subtract ``offset`` from a zarr array in place; False if it is absent."""
    if not array_path.is_dir():
        return False
    array = zarr.open_array(array_path, mode="r+")
    values = np.asarray(array[:])
    array[:] = (values.astype(np.float64) - offset).astype(values.dtype)
    return True


def _shift_table_coordinates(table_path: Path, shift: ProsegCoordinateShift) -> bool:
    """Subtract ``shift`` from a ProSeg AnnData table's centroids in place."""
    found_centroids = False
    for axis, column in _TABLE_CENTROID_COLUMNS.items():
        if _subtract_from_zarr_array(
            table_path / "obs" / column, float(getattr(shift, axis))
        ):
            found_centroids = True
    _subtract_from_zarr_array(
        table_path / "obsm" / "spatial", np.asarray([shift.x, shift.y])
    )
    return found_centroids


def restore_proseg_output_frame(
    zarr_path: str | Path, shift: ProsegCoordinateShift
) -> None:
    """Move every coordinate in a ProSeg SpatialData zarr back by ``shift``.

    ProSeg 3 writes transcript points (``x``, ``y``, ``observed_x``,
    ``observed_y``), cell polygons, and a cell table with ``centroid_x``,
    ``centroid_y`` and ``obsm["spatial"]``. Files are edited in place so the
    store keeps ProSeg's own layout. z is never shifted.

    Args:
        zarr_path: ProSeg ``--output-spatialdata`` store.
        shift: Offset that was added to the transcript coordinates.

    Raises:
        RuntimeError: If the store lacks transcript x/y, cell polygons, or
            cell centroids, since an unrecognised layout could otherwise keep
            shifted coordinates.
    """
    zarr_path = Path(zarr_path)
    if shift.is_identity:
        return
    offsets = {
        column: getattr(shift, axis)
        for axis, columns in _POINT_COLUMNS_BY_AXIS.items()
        for column in columns
    }
    shifted_point_columns: set[str] = set()
    for element in sorted((zarr_path / "points").glob("*/points.parquet")):
        parts = sorted(element.glob("*.parquet")) if element.is_dir() else [element]
        for part in parts:
            shifted_point_columns |= _shift_parquet_columns(part, offsets)
    shape_files = sorted((zarr_path / "shapes").glob("*/shapes.parquet"))
    for shape_file in shape_files:
        _shift_parquet_geometries(shape_file, shift)
    tables = sorted(path for path in (zarr_path / "tables").glob("*") if path.is_dir())
    has_centroids = [_shift_table_coordinates(table, shift) for table in tables]

    expected_point_columns = {
        column for column, offset in offsets.items() if offset != 0.0
    } & {"x", "y"}
    if not expected_point_columns <= shifted_point_columns:
        raise RuntimeError(
            f"ProSeg output {zarr_path} has no transcript "
            f"{sorted(expected_point_columns - shifted_point_columns)} column to "
            "restore after the coordinate shift."
        )
    if not shape_files or not any(has_centroids):
        raise RuntimeError(
            f"ProSeg output {zarr_path} lacks cell polygons or centroids; cannot "
            "restore them after the coordinate shift."
        )


def _parse_semver(version: str) -> tuple[int, int, int] | None:
    """Parse a semantic version string into integer components."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _check_proseg_available(proseg_binary: str | Path) -> str:
    """Verify ProSeg is executable and return its reported version."""
    try:
        result = subprocess.run(  # noqa: S603
            [str(proseg_binary), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        msg = (
            f"Proseg binary '{proseg_binary}' not found or not executable. "
            "Nextflow runs ENSURE_PROSEG automatically; for direct CLI use, "
            "install it with `cargo install proseg`."
        )
        raise RuntimeError(msg) from exc

    stdout = (result.stdout or "").strip()
    semver = _parse_semver(stdout)
    if semver is not None:
        return ".".join(str(x) for x in semver)
    parts = stdout.split()
    return parts[1] if len(parts) > 1 else "unknown"


def _check_proseg_version_gte(version: str, target: str) -> bool:
    """Return True when ``version`` is >= ``target``.

    If parsing fails, this function defaults to True to preserve compatibility
    with mixed binary outputs across ProSeg versions.
    """
    parsed_version = _parse_semver(version)
    parsed_target = _parse_semver(target)
    if parsed_version is None or parsed_target is None:
        logger.warning(
            "Could not parse ProSeg version '%s'; assuming it supports modern flags.",
            version,
        )
        return True
    return parsed_version >= parsed_target


def _validate_columns_from_csv(csv_path: Path, required_columns: list[str]) -> None:
    """Validate that required columns exist in an input CSV header."""
    header_columns = list(pd.read_csv(csv_path, nrows=0).columns)
    missing_columns = [col for col in required_columns if col not in header_columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in transcript CSV {csv_path}: {missing_columns}"
        )


def _to_optional_path_array(
    value: np.ndarray | str | Path | None,
    tmp_path: Path,
    filename: str,
    dtype: DTypeLike,
) -> Path | None:
    """Materialize array-like input to disk and return a path."""
    if value is None:
        return None
    if isinstance(value, str | Path):
        return Path(value)

    out_path = tmp_path / filename
    np.save(out_path, np.asarray(value, dtype=dtype))
    return out_path


def _publish_shifted_output(
    staged_path: Path,
    output_path: Path,
    shift: ProsegCoordinateShift,
    transforms: dict[str, list[float]],
) -> None:
    """Restore a staged ProSeg store to the input frame, then move it into place.

    The store only reaches ``output_path`` once every coordinate is restored,
    so a crash cannot leave a shifted store where later steps would reuse it.
    """
    try:
        restore_proseg_output_frame(staged_path, shift)
        record = {
            "description": (
                "MerXen added shift_um to transcript x/y and to the Cellpose "
                "mask-to-micron translations because the ProSeg command line "
                "cannot take negative numbers. Every ProSeg output coordinate "
                "was shifted back, so this store is in the input micron frame."
            ),
            "shift_um": {"x": shift.x, "y": shift.y},
            **transforms,
        }
        (staged_path / COORDINATE_SHIFT_SIDECAR).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        remove_path(output_path)
        os.replace(staged_path, output_path)
    except BaseException:
        remove_path(staged_path)
        raise


_PROSEG_ITER_RE = re.compile(r"[Ii]ter(?:ation)?\s*[:\s]\s*(\d+)(?:\s*/\s*(\d+))?")


def run_proseg_refinement(
    transcripts_df: pd.DataFrame | str | Path,
    output_path: str | Path,
    proseg_binary: str | Path,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    gene_col: str = "feature_name",
    cell_id_col: str = "cell_id",
    transcript_id_col: str | None = None,
    qv_col: str | None = None,
    samples: int = 1000,
    burnin_voxel_size: float | None = None,
    voxel_size: float = 0.5,
    voxel_layers: int = 1,
    nuclear_reassignment_prob: float = 0.2,
    diffusion_probability: float = 0.2,
    cell_compactness: float | None = None,
    expand_initialized_cells: int | None = None,
    use_cell_initialization: bool = False,
    prior_seg_reassignment_prob: float | None = None,
    morphology_steps_per_iter: int | None = None,
    max_transcript_nucleus_distance: float | None = None,
    cellpose_masks: np.ndarray | str | Path | None = None,
    cellpose_cellprobs: np.ndarray | str | Path | None = None,
    cellpose_scale: float | None = None,
    cellpose_x_transform: tuple[float, float, float] | None = None,
    cellpose_y_transform: tuple[float, float, float] | None = None,
    diffusion_sigma_far: float | None = None,
    num_threads: int = 12,
    overwrite: bool = True,
    logger: logging.Logger | None = None,
    progress_callback: Any = None,
    proseg_samples: int | None = None,
    temp_root: str | Path | None = None,
) -> Path:
    """Run ProSeg refinement on transcript data.

    Args:
        transcripts_df: Input transcript table or CSV path.
        output_path: Output SpatialData zarr path.
        proseg_binary: Path to the ProSeg executable.
        x_col: X coordinate column name.
        y_col: Y coordinate column name.
        z_col: Z coordinate column name.
        gene_col: Gene/feature column name.
        cell_id_col: Seed assignment column name.
        transcript_id_col: Optional stable numeric transcript identifier column.
        qv_col: Optional transcript quality-value column.
        samples: Number of MCMC samples.
        burnin_voxel_size: Optional burn-in voxel size (must be >= voxel_size).
        voxel_size: Voxel size for inference.
        voxel_layers: Number of voxel layers in z.
        nuclear_reassignment_prob: ProSeg nuclear reassignment probability.
        diffusion_probability: ProSeg diffusion probability.
        cell_compactness: Optional cell compactness prior.
        expand_initialized_cells: Optional cell expansion in voxels.
        use_cell_initialization: Whether to initialize from seeded assignments.
        prior_seg_reassignment_prob: Optional prior reassignment probability.
        morphology_steps_per_iter: Optional morphology sub-steps per iteration.
        max_transcript_nucleus_distance: Optional transcript distance cutoff.
        cellpose_masks: Optional Cellpose mask array or path.
        cellpose_cellprobs: Optional Cellpose probability array or path.
        cellpose_scale: Optional isotropic mask scale in microns per pixel.
        cellpose_x_transform: Optional affine x tuple (a, b, c).
        cellpose_y_transform: Optional affine y tuple (a, b, c).
        diffusion_sigma_far: Optional far-field diffusion sigma.
        num_threads: Number of CPU threads for ProSeg.
        overwrite: Whether to pass `--overwrite`.
        logger: Optional logger instance.
        temp_root: Directory for the temporary transcript CSV and mask arrays.
            Defaults to ``$MERXEN_TMPDIR`` or the working directory, never the
            system temporary directory (see ``tempdirs.large_temp_root``).

    Returns:
        Path to the produced output zarr.
    """
    log = logger or logging.getLogger(__name__)
    output_path = Path(output_path)
    proseg_binary = Path(proseg_binary)

    if int(samples) < 1:
        raise ValueError(f"samples must be >= 1 (got {samples})")
    if int(voxel_layers) < 1:
        raise ValueError(f"voxel_layers must be >= 1 (got {voxel_layers})")
    if int(voxel_layers) > 256:
        raise ValueError(f"voxel_layers exceeds ProSeg maximum of 256 ({voxel_layers})")
    if burnin_voxel_size is not None and burnin_voxel_size < voxel_size:
        raise ValueError(
            "burnin_voxel_size must be >= voxel_size "
            f"(got burnin_voxel_size={burnin_voxel_size}, voxel_size={voxel_size})"
        )

    required_columns = [x_col, y_col, z_col, gene_col, cell_id_col]
    if transcript_id_col is not None:
        required_columns.append(transcript_id_col)
    if qv_col is not None:
        required_columns.append(qv_col)
    transcript_csv_path: Path | None = None

    if isinstance(transcripts_df, str | Path):
        transcript_csv_path = Path(transcripts_df)
        if not transcript_csv_path.exists():
            raise FileNotFoundError(f"Transcript CSV not found: {transcript_csv_path}")
        _validate_columns_from_csv(transcript_csv_path, required_columns)
    else:
        missing_columns = [
            col for col in required_columns if col not in transcripts_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in transcripts_df: {missing_columns}"
            )

    proseg_version = _check_proseg_available(proseg_binary)
    use_zarr_output = _check_proseg_version_gte(proseg_version, "3.0.0")
    log.info("Using ProSeg version: %s", proseg_version)

    os.environ["RAYON_NUM_THREADS"] = str(int(num_threads))
    log.info("Using %s CPU threads for ProSeg", num_threads)

    proseg_x_transform = cellpose_x_transform
    proseg_y_transform = cellpose_y_transform
    shift = ProsegCoordinateShift()
    if cellpose_x_transform is not None and cellpose_y_transform is not None:
        if len(cellpose_x_transform) != 3 or len(cellpose_y_transform) != 3:
            raise ValueError(
                "cellpose_x_transform and cellpose_y_transform must be 3-tuples"
            )
        proseg_x_transform, proseg_y_transform, shift = proseg_safe_affine(
            cellpose_x_transform, cellpose_y_transform
        )
    proseg_output_path = output_path
    if not shift.is_identity:
        if not use_zarr_output:
            raise RuntimeError(
                "A negative mask-to-micron translation needs ProSeg >= 3.0.0 "
                "SpatialData output to undo the coordinate shift "
                f"(got {proseg_version})."
            )
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"ProSeg output already exists: {output_path}")
        proseg_output_path = output_path.parent / f".{output_path.name}.proseg-shifted"
        remove_path(proseg_output_path)
        log.warning(
            "Mask-to-micron affine has negative terms (x=%s, y=%s); running ProSeg "
            "with transcripts shifted by (%s, %s) um and shifting its outputs back",
            list(cellpose_x_transform or ()),
            list(cellpose_y_transform or ()),
            shift.x,
            shift.y,
        )

    with tempfile.TemporaryDirectory(
        prefix="merxen-proseg-", dir=large_temp_root(temp_root)
    ) as tmp_dir:
        tmp_path = Path(tmp_dir)

        if transcript_csv_path is None:
            if not isinstance(transcripts_df, pd.DataFrame):
                raise TypeError(
                    "transcripts_df must be a DataFrame when not using CSV input."
                )
            transcript_csv_path = tmp_path / "transcripts.csv"
            transcript_view = transcripts_df.loc[:, required_columns]
            transcript_view.to_csv(transcript_csv_path, index=False)
            log.info(
                "Saved %s transcripts to %s", len(transcript_view), transcript_csv_path
            )
        else:
            log.info("Using transcript CSV: %s", transcript_csv_path)
        if not shift.is_identity:
            shifted_csv_path = tmp_path / "transcripts_shifted.csv"
            _write_shifted_transcripts(
                transcript_csv_path,
                shifted_csv_path,
                required_columns,
                x_col,
                y_col,
                shift,
            )
            transcript_csv_path = shifted_csv_path

        cellpose_mask_path = _to_optional_path_array(
            value=cellpose_masks,
            tmp_path=tmp_path,
            filename="cellpose_masks.npy",
            dtype=np.uint32,
        )
        cellpose_cellprobs_path = _to_optional_path_array(
            value=cellpose_cellprobs,
            tmp_path=tmp_path,
            filename="cellpose_cellprobs.npy",
            dtype=np.float32,
        )

        if cellpose_mask_path is not None:
            has_scale = cellpose_scale is not None
            has_affine = (
                cellpose_x_transform is not None and cellpose_y_transform is not None
            )
            if not (has_scale or has_affine):
                raise ValueError(
                    "Cellpose masks were supplied but no transform was provided. "
                    "Provide cellpose_scale or both cellpose_x_transform/"
                    "cellpose_y_transform."
                )

        cmd = [
            str(proseg_binary),
            str(transcript_csv_path),
            "-x",
            x_col,
            "-y",
            y_col,
            "-z",
            z_col,
            "--gene-column",
            gene_col,
            "--cell-id-column",
            cell_id_col,
            "--cell-id-unassigned",
            "0",
            "--samples",
            str(int(samples)),
            "--voxel-size",
            str(float(voxel_size)),
            "--voxel-layers",
            str(int(voxel_layers)),
            "--nuclear-reassignment-prob",
            str(float(nuclear_reassignment_prob)),
            "--diffusion-probability",
            str(float(diffusion_probability)),
        ]

        if int(samples) < 100:
            # ProSeg's default recorded-samples=100 can fail for small sample counts.
            cmd.extend(["--recorded-samples", str(int(samples))])
        if transcript_id_col is not None:
            cmd.extend(["--transcript-id-column", transcript_id_col])
        if qv_col is not None:
            cmd.extend(["--qv-column", qv_col])
        if cell_compactness is not None:
            cmd.extend(["--cell-compactness", str(float(cell_compactness))])
        if expand_initialized_cells is not None:
            cmd.extend(
                ["--expand-initialized-cells", str(int(expand_initialized_cells))]
            )
        if use_cell_initialization:
            cmd.append("--use-cell-initialization")
        if prior_seg_reassignment_prob is not None:
            cmd.extend(
                [
                    "--prior-seg-reassignment-prob",
                    str(float(prior_seg_reassignment_prob)),
                ]
            )
        if morphology_steps_per_iter is not None:
            cmd.extend(
                ["--morphology-steps-per-iter", str(int(morphology_steps_per_iter))]
            )
        if max_transcript_nucleus_distance is not None:
            cmd.extend(
                [
                    "--max-transcript-nucleus-distance",
                    str(float(max_transcript_nucleus_distance)),
                ]
            )
        if burnin_voxel_size is not None:
            cmd.extend(["--burnin-voxel-size", str(float(burnin_voxel_size))])
        if diffusion_sigma_far is not None:
            cmd.extend(["--diffusion-sigma-far", str(float(diffusion_sigma_far))])
        if cellpose_mask_path is not None:
            cmd.extend(["--cellpose-masks", str(cellpose_mask_path)])
        if cellpose_cellprobs_path is not None:
            cmd.extend(["--cellpose-cellprobs", str(cellpose_cellprobs_path)])

        if proseg_x_transform is not None and proseg_y_transform is not None:
            cmd.extend(
                ["--cellpose-x-transform"] + [str(x) for x in proseg_x_transform]
            )
            cmd.extend(
                ["--cellpose-y-transform"] + [str(y) for y in proseg_y_transform]
            )
        elif cellpose_scale is not None:
            cmd.extend(["--cellpose-scale", str(float(cellpose_scale))])

        if use_zarr_output:
            cmd.extend(["--output-spatialdata", str(proseg_output_path)])
        if overwrite:
            cmd.append("--overwrite")

        log.info("Running ProSeg command: %s", " ".join(cmd))
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        streamed_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            streamed_lines.append(line)
            log.info("[proseg] %s", line)
            if progress_callback is not None:
                m = _PROSEG_ITER_RE.search(line)
                if m:
                    done = int(m.group(1))
                    total = int(m.group(2)) if m.group(2) else proseg_samples
                    pct = round(100 * done / total, 1) if total else None
                    progress_callback(
                        "proseg_sampling",
                        samples_done=done,
                        samples_total=total,
                        pct=pct,
                    )

        return_code = process.wait()
        if return_code != 0:
            tail = "\n".join(streamed_lines[-100:])
            if tail:
                log.error("Last ProSeg output lines:\n%s", tail)
            if proseg_output_path != output_path:
                remove_path(proseg_output_path)
            raise RuntimeError(
                f"ProSeg execution failed with return code {return_code}"
            )

    if proseg_output_path != output_path:
        if not proseg_output_path.exists():
            raise RuntimeError(f"ProSeg output not found at {proseg_output_path}")
        _publish_shifted_output(
            proseg_output_path,
            output_path,
            shift,
            {
                "input_x_transform": list(cellpose_x_transform or ()),
                "input_y_transform": list(cellpose_y_transform or ()),
                "proseg_x_transform": list(proseg_x_transform or ()),
                "proseg_y_transform": list(proseg_y_transform or ()),
            },
        )
        log.info("Restored ProSeg outputs to the input micron frame")

    if not output_path.exists():
        raise RuntimeError(f"ProSeg output not found at {output_path}")

    log.info("ProSeg completed successfully: %s", output_path)
    return output_path
