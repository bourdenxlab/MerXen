"""Export the inputs of one ATERA ProSeg tile (inside ATERA_PROSEG_TILE).

Rows are the transcripts of the latest/source store whose pixel
``(floor(x / ps), floor(y / ps))`` (the vendor's float32 corner rule) lies in
the tile window, that are genes and have ``qv >= min_qv``. They are written in
ProSeg's Xenium parquet layout:

========================  ==========================================
column                    content
========================  ==========================================
``transcript_id``         uint64, the store's
``x_location`` ...        float32 microns (``z_location``, ``qv`` too)
``feature_name``          plain string
``cell_id``               string vendor label (``original_assignment``),
                          ``"0"`` when unassigned
``overlaps_nucleus``      uint8
``fov_name``              constant ``"0"`` (ProSeg keys cells by fov)
========================  ==========================================

Beside the parquet, the transcript ids of every source file are kept
(``exported/<file>.npy``, sorted) so the claims can be written per source
file, and the vendor cell mask of the window is saved as the uint32 prior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from merxen.enrichment.label_lookup import PixelFrame, pixel_indices
from merxen.segmentation.atera_proseg import AteraProsegError

logger = logging.getLogger(__name__)

POINTS_KEY = "transcripts"
SOURCE_COLUMNS = (
    "x",
    "y",
    "z",
    "feature_name",
    "is_gene",
    "qv",
    "overlaps_nucleus",
    "original_assignment",
    "transcript_id",
)
XENIUM_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.uint64()),
        ("x_location", pa.float32()),
        ("y_location", pa.float32()),
        ("z_location", pa.float32()),
        ("feature_name", pa.string()),
        ("cell_id", pa.string()),
        ("overlaps_nucleus", pa.uint8()),
        ("qv", pa.float32()),
        ("fov_name", pa.string()),
    ]
)
"""Arrow schema of the ProSeg input (ProSeg 3.2.0 ``xenium`` parquet reader)."""

UNASSIGNED_CELL_ID = "0"
FOV_NAME = "0"


@dataclass
class TileExport:
    """What :func:`export_tile` wrote.

    Attributes:
        parquet_path: The ProSeg input.
        prior_path: The uint32 vendor mask of the window.
        exported_dir: Folder of the per-source-file transcript ids.
        n_rows: Rows written.
        n_rows_assigned: Rows with a vendor cell.
        n_labels_in_window: Distinct non-zero labels of the prior.
        source_files: Source parquet file names with exported rows, in
            order, with their row counts.
        n_files_scanned: Source files read (after footer pruning).
        n_genes: Distinct genes written.
    """

    parquet_path: Path
    prior_path: Path
    exported_dir: Path
    n_rows: int = 0
    n_rows_assigned: int = 0
    n_labels_in_window: int = 0
    source_files: dict[str, int] = field(default_factory=dict)
    n_files_scanned: int = 0
    n_genes: int = 0


def window_bbox_um(
    window_px: list[int] | tuple[int, ...], pixel_size_um: float
) -> tuple[float, float, float, float]:
    """Return the ``(x0, y0, x1, y1)`` microns of a pixel window."""
    col0, col1, row0, row1 = (int(v) for v in window_px)
    return (
        col0 * pixel_size_um,
        row0 * pixel_size_um,
        col1 * pixel_size_um,
        row1 * pixel_size_um,
    )


def window_row_mask(
    x: np.ndarray, y: np.ndarray, window_px: list[int] | tuple[int, ...], ps: float
) -> np.ndarray:
    """Return which coordinates fall on a pixel of the window (corner rule).

    Args:
        x: X microns.
        y: Y microns.
        window_px: ``[col0, col1, row0, row1]`` (end-exclusive).
        ps: Microns per pixel.

    Returns:
        A boolean mask.
    """
    col0, col1, row0, row1 = (int(v) for v in window_px)
    column, row = pixel_indices(x, y, PixelFrame(um_per_px=ps), convention="corner")
    return (column >= col0) & (column < col1) & (row >= row0) & (row < row1)


def _plain_strings(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """Return a plain (non-dictionary) string array."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    return pc.cast(array, pa.string())


def _constant_strings(value: str, n_rows: int) -> pa.Array:
    indices = pa.array(np.zeros(n_rows, dtype=np.int8))
    return pa.DictionaryArray.from_arrays(
        indices, pa.array([value])
    ).dictionary_decode()


def xenium_layout_table(
    table: pa.Table, *, nuclear_compartment: bool = True
) -> pa.Table:
    """Convert filtered store rows to ProSeg's Xenium parquet layout.

    Args:
        table: Rows with the store columns (:data:`SOURCE_COLUMNS`).
        nuclear_compartment: Keep ``overlaps_nucleus``; ``False`` writes 0.

    Returns:
        A table with :data:`XENIUM_SCHEMA`.
    """
    n_rows = table.num_rows
    assignment = table.column("original_assignment").combine_chunks()
    cell_id = pc.fill_null(pc.cast(assignment, pa.string()), UNASSIGNED_CELL_ID)
    if nuclear_compartment:
        nucleus = pc.cast(
            pc.fill_null(table.column("overlaps_nucleus").combine_chunks(), False),
            pa.uint8(),
        )
    else:
        nucleus = pa.array(np.zeros(n_rows, dtype=np.uint8))
    return pa.Table.from_arrays(
        [
            pc.cast(table.column("transcript_id"), pa.uint64()).combine_chunks(),
            pc.cast(table.column("x"), pa.float32()).combine_chunks(),
            pc.cast(table.column("y"), pa.float32()).combine_chunks(),
            pc.cast(table.column("z"), pa.float32()).combine_chunks(),
            _plain_strings(table.column("feature_name")),
            cell_id,
            nucleus,
            pc.cast(table.column("qv"), pa.float32()).combine_chunks(),
            _constant_strings(FOV_NAME, n_rows),
        ],
        schema=XENIUM_SCHEMA,
    )


def _prior(store: Path, window_px: list[int] | tuple[int, ...]) -> np.ndarray:
    from merxen.segmentation.atera_proseg.plan import _level0

    col0, col1, row0, row1 = (int(v) for v in window_px)
    level0 = _level0(Path(store) / "labels" / "cell_labels")
    return np.ascontiguousarray(level0[row0:row1, col0:col1], dtype=np.uint32)


def export_tile(
    store: Path,
    window_px: list[int] | tuple[int, ...],
    pixel_size_um: float,
    output_dir: Path,
    *,
    min_qv: float,
    nuclear_compartment: bool = True,
) -> TileExport:
    """Write the ProSeg input, the prior and the exported-id side files.

    Args:
        store: The store whose points and vendor labels are read.
        window_px: ``[col0, col1, row0, row1]``.
        pixel_size_um: Microns per pixel.
        output_dir: Folder for ``tile.parquet``, ``prior.npy`` and
            ``exported/``.
        min_qv: Smallest QV exported. The build already applies it, so a
            gene row below it is logged as a warning and not exported.
        nuclear_compartment: Give ProSeg ``overlaps_nucleus``.

    Returns:
        The export record.

    Raises:
        AteraProsegError: If the window holds no gene row.
    """
    from merxen.visualization.raster_density import points_files

    output_dir = Path(output_dir)
    exported_dir = output_dir / "exported"
    exported_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "tile.parquet"
    prior_path = output_dir / "prior.npy"
    bbox = window_bbox_um(window_px, pixel_size_um)
    files, n_total, pruned = points_files(Path(store), POINTS_KEY, bbox)
    record = TileExport(
        parquet_path=parquet_path, prior_path=prior_path, exported_dir=exported_dir
    )
    record.n_files_scanned = len(files)
    genes: set[str] = set()
    writer = pq.ParquetWriter(parquet_path, XENIUM_SCHEMA, compression="zstd")
    try:
        for path in files:
            table = pq.read_table(path, columns=list(SOURCE_COLUMNS))
            x = table.column("x").to_numpy()
            y = table.column("y").to_numpy()
            keep = window_row_mask(x, y, window_px, pixel_size_um)
            keep &= table.column("is_gene").to_numpy(zero_copy_only=False).astype(bool)
            qv = table.column("qv").to_numpy()
            low = keep & ~(qv >= min_qv)
            n_low = int(low.sum())
            if n_low:
                logger.warning(
                    "%s: %d gene rows in the window have qv < %s; not exported",
                    path.name,
                    n_low,
                    min_qv,
                )
                keep &= ~low
            if not keep.any():
                continue
            rows = table.filter(pa.array(keep))
            out = xenium_layout_table(rows, nuclear_compartment=nuclear_compartment)
            writer.write_table(out)
            ids = np.sort(out.column("transcript_id").to_numpy())
            np.save(exported_dir / f"{path.name}.npy", ids)
            record.source_files[path.name] = int(out.num_rows)
            record.n_rows += int(out.num_rows)
            record.n_rows_assigned += int(
                pc.sum(pc.not_equal(out.column("cell_id"), UNASSIGNED_CELL_ID)).as_py()
                or 0
            )
            genes.update(pc.unique(out.column("feature_name")).to_pylist())
    finally:
        writer.close()
    if record.n_rows == 0:
        raise AteraProsegError("The tile window holds no gene transcript")
    record.n_genes = len(genes)
    prior = _prior(Path(store), window_px)
    np.save(prior_path, prior)
    labels = np.unique(prior)
    record.n_labels_in_window = int((labels != 0).sum())
    del prior
    logger.info(
        "Exported %d rows (%d assigned, %d genes) from %d of %d files%s; "
        "%d labels in the window",
        record.n_rows,
        record.n_rows_assigned,
        record.n_genes,
        len(files),
        n_total,
        " (footer-pruned)" if pruned else "",
        record.n_labels_in_window,
    )
    return record


def read_exported_ids(exported_dir: Path) -> dict[str, np.ndarray]:
    """Return ``source file name -> sorted exported transcript ids``."""
    found: dict[str, Any] = {}
    for path in sorted(Path(exported_dir).glob("*.npy")):
        found[path.name.removesuffix(".npy")] = np.load(path)
    return found
