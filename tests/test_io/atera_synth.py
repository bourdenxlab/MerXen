"""Tiny synthetic 10x Atera bundle that reproduces the real output format.

:func:`write_atera_bundle` writes one ROI directory into a test's ``tmp_path``:
zipped zarr v2 stores (``transcripts.zarr.zip``, ``cells.zarr.zip``,
``cell_feature_matrix.zarr.zip``; every member STORED), a multi-file pyramidal
JPEG-2000 OME-TIFF ``morphology_2d/`` with OME ``TiffData`` file-to-channel
mapping, ``experiment.spatial``, ``panel_config.json``, a metrics summary, the
vendor reports, the FOV locations, a vendor DE table and the H&E alignment
companions. Nothing binary is committed.

The geometry is fixed and small (97 x 67 pixels at the real pixel size). The
first five cells cover the special cases of the real data:

- cell 0: one nucleus;
- cell 1: two nuclei, the second one larger (``nucleus_area`` = largest);
- cell 2: no nucleus (NaN nucleus fields in ``cell_summary``);
- cell 3: one label made of two connected components;
- cell 4: its nucleus polygon has ``num_vertices == 0``.

Cells from the sixth on sit at x >= 10 microns (tile ``1,0`` is empty by
default), so they have no transcripts and ``filtered=False`` in the
cell-feature matrix.

Transcripts follow the vendor rules: rows sorted by gene, then codeword, in
each tile; an end-exclusive ``gene_offset``; a transcript is assigned to a
cell exactly when its QV is at least 20 and its floor pixel lies on a
``masks/1`` label; unassigned rows carry ``(0xFFFFFFFF, 0xFFFFFFFF)``. The
identifying manifest fields hold obviously fake values (:data:`FAKE_IDENTIFIERS`).
"""

from __future__ import annotations

import json
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anndata as ad
import numcodecs
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import sparse

PIXEL_SIZE_UM: Final = 0.21249222
GRID_SIZE_UM: Final = 10.0
IMAGE_HEIGHT: Final = 97
EXTRA_CELL_START_COL: Final = 50
SLOT_PX: Final = 14
UNASSIGNED: Final = 0xFFFFFFFF
DEFAULT_TILES: Final = {"0,0": 40, "-1,1": 3, "1,0": 0, "0,1": 25}
DEFAULT_GRID_KEYS_ORDER: Final = ("0,1", "-1,1", "0,0", "1,0")
CATEGORY_COLUMNS: Final = (
    "is_gene",
    "is_negative_control_codeword",
    "is_negative_control_probe",
    "is_unassigned_codeword",
    "is_predesigned_gene",
    "is_deprecated_codeword",
    "is_custom_gene",
    "is_genomic_control",
    "is_boosted_gene",
)

_GENE: Final = "Gene Expression"
_PREDESIGNED: Final = ("is_gene", "is_predesigned_gene")
#: (symbol, feature_type, category flags, feature_id, codeword count)
FEATURES: Final = (
    ("SNAP25", _GENE, _PREDESIGNED, "ENSG00000132639", 3),
    ("GAD1", _GENE, _PREDESIGNED, "ENSG00000128683", 2),
    ("MBP", _GENE, _PREDESIGNED, "ENSG00000197971", 3),
    ("ACTB", _GENE, _PREDESIGNED, "ENSG00000075624", 2),
    ("ENSG00000289701", _GENE, _PREDESIGNED, "ENSG00000289701", 2),
    ("BOOSTGENE1", _GENE, ("is_gene", "is_boosted_gene"), "ENSG00000111111", 2),
    ("CUSTOMGENE1", _GENE, ("is_gene", "is_custom_gene"), "ENSG00000222222", 2),
    (
        "NegControlCodeword_0500",
        "Negative Control Codeword",
        ("is_negative_control_codeword",),
        "NegControlCodeword_0500",
        1,
    ),
    (
        "NegControlProbe_00042",
        "Negative Control Probe",
        ("is_negative_control_probe",),
        "NegControlProbe_00042",
        1,
    ),
    (
        "UnassignedCodeword_0007",
        "Unassigned Codeword",
        ("is_unassigned_codeword",),
        "UnassignedCodeword_0007",
        1,
    ),
    (
        "Intergenic_Region_12",
        "Genomic Control",
        ("is_genomic_control",),
        "Intergenic_Region_12",
        1,
    ),
    (
        "DeprecatedCodeword_0003",
        "Deprecated Codeword",
        ("is_deprecated_codeword",),
        "DeprecatedCodeword_0003",
        1,
    ),
)

#: Identifying manifest values of the fixture, obviously fake: the store keeps
#: them verbatim, so tests can check the metadata is complete.
FAKE_IDENTIFIERS: Final = {
    "run_name": "SYNTH_RUN_ZZ",
    "slide_name": "SYNTHSLIDE",
    "slide_id": "SYNTHSLIDEID",
    "region_name": "SYNTH_REGION_ZZ",
    "instrument_sn": "SYNTHSN000",
    "analysis_uuid": "deadbeef-0000-4000-8000-000000000001",
    "experiment_uuid": "deadbeef-0000-4000-8000-000000000002",
    "cassette_uuid": "deadbeef-0000-4000-8000-000000000003",
    "roi_uuid": "deadbeef-0000-4000-8000-000000000004",
    "well_uuid": "deadbeef-0000-4000-8000-000000000005",
    "calibration_uuid": "deadbeef-0000-4000-8000-000000000006",
    "customer_email": "synthetic.customer@example.invalid",
    "customer_name": "Synthetic Customer",
    "designer_email": "synthetic.designer@example.invalid",
    "designer_name": "Synthetic Designer",
    "dataset_uuid": "deadbeef-0000-4000-8000-000000000007",
    "run_start_time": "1999-12-31T23:59:58Z",
    "reagent_lot": "SYNTHLOT0001",
    "panel_label": "Synthetic customer panel label",
}
RUN_SUFFIX: Final = (
    f"{FAKE_IDENTIFIERS['instrument_sn']}_{FAKE_IDENTIFIERS['slide_name']}_"
    f"{FAKE_IDENTIFIERS['region_name']}"
)
COMPANION_PREFIX: Final = "Synthetic_Atera_dataset"

CHANNELS_2: Final = (("DAPI", "dapi"), ("18S", "18s"))
CHANNELS_4: Final = (
    ("DAPI", "dapi"),
    ("ATP1A1/CD45/E-Cadherin", "atp1a1_cd45_e-cadherin"),
    ("18S", "18s"),
    ("alphaSMA/Vimentin", "alphasma_vimentin"),
)
SEGMENTATION_METHODS: Final = (
    "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)",
    "Segmented by interior stain (18S)",
    "Segmented by nucleus expansion of 5.0µm",
    "Segmented by nuclear stain (DAPI) multi-focus",
    "Segmented by nuclear stain (DAPI) z stack",
    "Segmented by boundary stain at 0.25x (ATP1A1+CD45+E-Cadherin)",
    "Imported Cell Segmentation",
    "Imported Nucleus Segmentation",
)
N_STORED_LEVELS: Final = 5
TIFF_TILE: Final = (16, 16)

# Pixel rectangles (row_start, row_stop, col_start, col_stop) of the special cells.
_SPECIAL_CELLS: Final = (
    ((6, 18, 6, 20),),
    ((6, 21, 26, 45),),
    ((26, 36, 8, 20),),
    ((52, 62, 6, 16), (52, 62, 22, 30)),
    ((70, 84, 10, 26),),
)
_SPECIAL_NUCLEI: Final = (
    ((9, 14, 9, 15),),
    ((8, 13, 28, 33), (12, 19, 36, 43)),
    (),
    ((54, 60, 8, 14),),
    ((74, 80, 14, 22),),
)
TWO_NUCLEUS_CELL: Final = 1
NUCLEUS_LESS_CELL: Final = 2
MULTI_COMPONENT_CELL: Final = 3
EMPTY_POLYGON_CELL: Final = 4


@dataclass(frozen=True)
class AteraSynth:
    """What :func:`write_atera_bundle` wrote, with the ground truth.

    Attributes:
        root: The ROI directory.
        pixel_size_um: Microns per morphology pixel.
        gene_names: Feature names in vendor order (genes, then controls).
        feature_ids: CFM ``var`` index (ENSG for genes, the name for controls).
        feature_types: CFM ``feature_type`` of each feature.
        is_gene: Whether each feature is a gene.
        codeword_gene_mapping: Gene index of each codeword.
        grid_keys_order: Tile keys in ``grid_keys`` attribute order.
        tile_rows: Row count of each tile.
        transcripts: Every transcript row in vendor ordinal order, with
            columns ``tile``, ``row``, ``x``, ``y``, ``z``, ``gene``,
            ``codeword``, ``qv``, ``prefix``, ``suffix``, ``overlaps_nucleus``,
            ``cell_row`` (-1 if unassigned) and ``transcript_id``.
        cell_prefix: Cell ID prefix per cell row.
        cell_suffix: Dataset suffix per cell row.
        cell_rects: Pixel rectangles of each cell.
        cell_mask: ``masks/1``.
        nucleus_mask: ``masks/0``.
        nucleus_cell_index: Owning cell row of each nucleus row.
        cfm_counts: Dense ``cells x features`` vendor counts.
        cfm_filtered: CFM ``obs.filtered`` (True = kept).
        channel_names: Morphology channel names in OME order.
        channel_files: File of each channel, in OME order.
        channel_level0: Level-0 pixels of each channel, in OME order.
        companion_prefix: Prefix of the zip companions, if prefixed.
    """

    root: Path
    pixel_size_um: float
    gene_names: tuple[str, ...]
    feature_ids: tuple[str, ...]
    feature_types: tuple[str, ...]
    is_gene: np.ndarray
    codeword_gene_mapping: np.ndarray
    grid_keys_order: tuple[str, ...]
    tile_rows: dict[str, int]
    transcripts: pd.DataFrame
    cell_prefix: np.ndarray
    cell_suffix: np.ndarray
    cell_rects: tuple[tuple[tuple[int, int, int, int], ...], ...]
    cell_mask: np.ndarray
    nucleus_mask: np.ndarray
    nucleus_cell_index: np.ndarray
    cfm_counts: np.ndarray
    cfm_filtered: np.ndarray
    channel_names: tuple[str, ...]
    channel_files: tuple[str, ...]
    channel_level0: tuple[np.ndarray, ...]
    companion_prefix: str | None

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.cell_prefix)

    def zip_path(self, name: str) -> Path:
        """Path of one zarr zip, e.g. ``transcripts`` or ``cell_feature_matrix``."""
        prefix = f"{self.companion_prefix}_" if self.companion_prefix else ""
        return self.root / f"{prefix}{name}.zarr.zip"


def write_atera_bundle(
    root: Path,
    *,
    tiles: Mapping[str, int] | None = None,
    grid_keys_order: Sequence[str] | None = None,
    n_cells: int = 6,
    unassigned_fraction: float = 0.25,
    low_qv_fraction: float = 0.2,
    n_channels: int = 2,
    file_order_differs: bool = False,
    with_is_sketch: bool = False,
    prefixed_companions: bool = False,
    shuffle_category_columns: bool = False,
    rna_major_version: int = 6,
    cells_major_version: int = 6,
    analysis_sw_version: str = "atera-1.1.1.0",
    organism: str = "Human",
    seed: int = 0,
) -> AteraSynth:
    """Write a synthetic Atera ROI directory.

    Args:
        root: Directory to create (must not exist or be empty).
        tiles: Row count per tile key ``"x,y"``; tile ``i,j`` holds
            ``x in [10 i, 10 (i + 1))`` microns. Defaults to
            :data:`DEFAULT_TILES` (unequal sizes, a negative key, an empty tile).
        grid_keys_order: Order of the ``grid_keys`` attribute; must name every
            tile once. Defaults to :data:`DEFAULT_GRID_KEYS_ORDER` (not
            lexicographic) for the default tiles, else the order of ``tiles``.
        n_cells: Number of cells, at least 5.
        unassigned_fraction: Share of rows placed anywhere in their tile
            rather than on a cell pixel.
        low_qv_fraction: Share of rows with QV below 20 (never assigned).
        n_channels: Morphology channels, 2 (DAPI, 18S) or 4.
        file_order_differs: Whether file names order the channels in reverse
            of the OME ``TiffData`` order.
        with_is_sketch: Whether the CFM ``obs`` has ``is_sketch``.
        prefixed_companions: Whether the zarr zips carry a dataset prefix that
            ``experiment.spatial`` does not name (the public download layout).
        shuffle_category_columns: Whether ``gene_category`` and
            ``codeword_category`` use a non-default column order.
        rna_major_version: ``RnaDataset`` major version.
        cells_major_version: ``CellSegmentationDataset`` major version.
        analysis_sw_version: ``experiment.spatial`` analysis version.
        organism: ``panel_wta.panel_organism``.
        seed: Random seed.

    Returns:
        The written bundle and its ground truth.

    Raises:
        ValueError: If the arguments are inconsistent.
    """
    if grid_keys_order is None:
        grid_keys_order = DEFAULT_GRID_KEYS_ORDER if tiles is None else tuple(tiles)
    tiles = dict(DEFAULT_TILES if tiles is None else tiles)
    order = tuple(grid_keys_order)
    if sorted(order) != sorted(tiles):
        raise ValueError("grid_keys_order must name every tile exactly once")
    if n_cells < 5:
        raise ValueError("n_cells must be at least 5")
    if n_channels not in (2, 4):
        raise ValueError("n_channels must be 2 or 4")
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    rects, nuclei = _cell_layout(n_cells)
    width = EXTRA_CELL_START_COL + SLOT_PX * _extra_columns(n_cells) + 3
    cell_mask, nucleus_mask, nucleus_cell_index = _paint_masks(
        rects, nuclei, (IMAGE_HEIGHT, width)
    )
    cell_prefix = np.sort(
        rng.choice(np.arange(59_979, 3_905_076_576, 7919), n_cells, replace=False)
    ).astype(np.uint32)
    cell_suffix = np.ones(n_cells, dtype=np.uint32)

    gene_names = tuple(feature[0] for feature in FEATURES)
    codeword_gene_mapping = np.repeat(
        np.arange(len(FEATURES)), [feature[4] for feature in FEATURES]
    )
    category = np.zeros((len(FEATURES), len(CATEGORY_COLUMNS)), dtype=bool)
    for gene, feature in enumerate(FEATURES):
        for flag in feature[2]:
            category[gene, CATEGORY_COLUMNS.index(flag)] = True

    transcripts = _make_transcripts(
        rng,
        tiles,
        order,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        cell_prefix=cell_prefix,
        cell_suffix=cell_suffix,
        codeword_gene_mapping=codeword_gene_mapping,
        unassigned_fraction=unassigned_fraction,
        low_qv_fraction=low_qv_fraction,
    )
    is_gene = category[:, CATEGORY_COLUMNS.index("is_gene")]
    cfm_counts = np.zeros((n_cells, len(FEATURES)), dtype=np.int64)
    assigned = transcripts["cell_row"].to_numpy() >= 0
    np.add.at(
        cfm_counts,
        (
            transcripts["cell_row"].to_numpy()[assigned],
            transcripts["gene"].to_numpy()[assigned],
        ),
        1,
    )

    prefix = COMPANION_PREFIX if prefixed_companions else None
    zip_prefix = f"{prefix}_" if prefix else ""
    columns = (
        tuple(reversed(CATEGORY_COLUMNS))
        if shuffle_category_columns
        else CATEGORY_COLUMNS
    )
    column_index = [CATEGORY_COLUMNS.index(name) for name in columns]
    _write_transcripts_zip(
        root / f"{zip_prefix}transcripts.zarr.zip",
        transcripts,
        tiles=tiles,
        order=order,
        gene_names=gene_names,
        codeword_gene_mapping=codeword_gene_mapping,
        category=category[:, column_index],
        category_columns=columns,
        major_version=rna_major_version,
    )
    summary = _cell_summary(cell_mask, nucleus_mask, nucleus_cell_index, n_cells)
    _write_cells_zip(
        root / f"{zip_prefix}cells.zarr.zip",
        rects=rects,
        nuclei=nuclei,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        nucleus_cell_index=nucleus_cell_index,
        cell_prefix=cell_prefix,
        cell_suffix=cell_suffix,
        summary=summary,
        major_version=cells_major_version,
    )
    cfm_filtered = cfm_counts.sum(axis=1) >= 1
    _write_cfm_zip(
        root / f"{zip_prefix}cell_feature_matrix.zarr.zip",
        rng,
        counts=cfm_counts,
        filtered=cfm_filtered,
        summary=summary,
        cell_prefix=cell_prefix,
        cell_suffix=cell_suffix,
        with_is_sketch=with_is_sketch,
    )
    channels = CHANNELS_2 if n_channels == 2 else CHANNELS_4
    channel_files, channel_level0 = _write_morphology(
        root / "morphology_2d",
        rng,
        channels=channels,
        shape=(IMAGE_HEIGHT, width),
        file_order_differs=file_order_differs,
    )
    _write_small_files(
        root,
        organism=organism,
        analysis_sw_version=analysis_sw_version,
        n_cells=n_cells,
        n_transcripts=len(transcripts),
        four_channels=n_channels == 4,
    )
    return AteraSynth(
        root=root,
        pixel_size_um=PIXEL_SIZE_UM,
        gene_names=gene_names,
        feature_ids=tuple(feature[3] for feature in FEATURES),
        feature_types=tuple(feature[1] for feature in FEATURES),
        is_gene=is_gene,
        codeword_gene_mapping=codeword_gene_mapping,
        grid_keys_order=order,
        tile_rows=tiles,
        transcripts=transcripts,
        cell_prefix=cell_prefix,
        cell_suffix=cell_suffix,
        cell_rects=tuple(rects),
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        nucleus_cell_index=nucleus_cell_index,
        cfm_counts=cfm_counts,
        cfm_filtered=cfm_filtered,
        channel_names=tuple(name for name, _ in channels),
        channel_files=channel_files,
        channel_level0=channel_level0,
        companion_prefix=prefix,
    )


def cell_id_strings(prefixes: np.ndarray, suffixes: np.ndarray) -> list[str]:
    """Return the 10x string form of cell IDs, computed independently of src."""
    shift = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    return [
        format(int(p), "x").translate(shift).rjust(8, "a") + f"-{int(s)}"
        for p, s in zip(prefixes, suffixes, strict=True)
    ]


def _extra_columns(n_cells: int) -> int:
    rows_per_column = (IMAGE_HEIGHT - 4) // SLOT_PX
    return max(1, -(-(n_cells - len(_SPECIAL_CELLS)) // rows_per_column))


def _cell_layout(
    n_cells: int,
) -> tuple[
    list[tuple[tuple[int, int, int, int], ...]],
    list[tuple[tuple[int, int, int, int], ...]],
]:
    rects = list(_SPECIAL_CELLS)
    nuclei = list(_SPECIAL_NUCLEI)
    rows_per_column = (IMAGE_HEIGHT - 4) // SLOT_PX
    for extra in range(n_cells - len(_SPECIAL_CELLS)):
        column, row = divmod(extra, rows_per_column)
        r0 = 3 + row * SLOT_PX
        c0 = EXTRA_CELL_START_COL + column * SLOT_PX
        rects.append(((r0, r0 + 11, c0, c0 + 10),))
        nuclei.append(((r0 + 3, r0 + 8, c0 + 3, c0 + 7),))
    return rects, nuclei


def _paint_masks(
    rects: Sequence[tuple[tuple[int, int, int, int], ...]],
    nuclei: Sequence[tuple[tuple[int, int, int, int], ...]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_mask = np.zeros(shape, dtype=np.uint32)
    nucleus_mask = np.zeros(shape, dtype=np.uint32)
    owners: list[int] = []
    for cell, (cell_rects, cell_nuclei) in enumerate(zip(rects, nuclei, strict=True)):
        for r0, r1, c0, c1 in cell_rects:
            cell_mask[r0:r1, c0:c1] = cell + 1
        for r0, r1, c0, c1 in cell_nuclei:
            owners.append(cell)
            nucleus_mask[r0:r1, c0:c1] = len(owners)
    return cell_mask, nucleus_mask, np.asarray(owners, dtype=np.uint32)


def _snap(values: np.ndarray) -> np.ndarray:
    """Round microns to multiples of 1/64, as the vendor stores them."""
    return np.round(values * 64.0) / 64.0


def _sample_on_pixel(
    rng: np.random.Generator, row: int, col: int, bounds: tuple[float, ...]
) -> tuple[float, float] | None:
    """Place a point well inside one pixel and its tile, or return None."""
    for _ in range(50):
        x = float(_snap(np.asarray((col + rng.uniform(0.15, 0.85)) * PIXEL_SIZE_UM)))
        y = float(_snap(np.asarray((row + rng.uniform(0.15, 0.85)) * PIXEL_SIZE_UM)))
        fx, fy = x / PIXEL_SIZE_UM - col, y / PIXEL_SIZE_UM - row
        inside = bounds[0] <= x < bounds[1] and bounds[2] <= y < bounds[3]
        if 0.05 < fx < 0.95 and 0.05 < fy < 0.95 and inside:
            return x, y
    return None


def _make_transcripts(
    rng: np.random.Generator,
    tiles: Mapping[str, int],
    order: Sequence[str],
    *,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_prefix: np.ndarray,
    cell_suffix: np.ndarray,
    codeword_gene_mapping: np.ndarray,
    unassigned_fraction: float,
    low_qv_fraction: float,
) -> pd.DataFrame:
    n_features = int(codeword_gene_mapping.max()) + 1
    height, width = cell_mask.shape
    qv_low = np.arange(8.0, 20.0, 0.25)
    qv_high = np.arange(20.0, 40.25, 0.25)
    frames = []
    offset = 0
    first_tile = max(tiles, key=lambda key: tiles[key])
    for key in order:
        n_rows = int(tiles[key])
        x_index, y_index = (int(part) for part in key.split(","))
        bounds = (
            x_index * GRID_SIZE_UM,
            (x_index + 1) * GRID_SIZE_UM,
            y_index * GRID_SIZE_UM,
            (y_index + 1) * GRID_SIZE_UM,
        )
        rows_px, cols_px = np.nonzero(cell_mask)
        centre_x = (cols_px + 0.5) * PIXEL_SIZE_UM
        centre_y = (rows_px + 0.5) * PIXEL_SIZE_UM
        in_tile = (
            (centre_x >= bounds[0] + 0.2)
            & (centre_x < bounds[1] - 0.2)
            & (centre_y >= bounds[2] + 0.2)
            & (centre_y < bounds[3] - 0.2)
        )
        cell_pixels = np.column_stack([rows_px[in_tile], cols_px[in_tile]])
        genes = rng.integers(0, n_features, n_rows)
        if key == first_tile:
            # Every feature appears at least once in the largest tile.
            genes[: min(n_rows, n_features)] = np.arange(min(n_rows, n_features))
        xs = np.empty(n_rows)
        ys = np.empty(n_rows)
        for row in range(n_rows):
            placed = None
            if len(cell_pixels) and rng.random() >= unassigned_fraction:
                pixel_row, pixel_col = cell_pixels[rng.integers(len(cell_pixels))]
                placed = _sample_on_pixel(rng, int(pixel_row), int(pixel_col), bounds)
            while placed is None:
                x = float(_snap(np.asarray(rng.uniform(bounds[0], bounds[1]))))
                y = float(_snap(np.asarray(rng.uniform(bounds[2], bounds[3]))))
                fx = (x / PIXEL_SIZE_UM) % 1.0
                fy = (y / PIXEL_SIZE_UM) % 1.0
                if 0.05 < fx < 0.95 and 0.05 < fy < 0.95 and x < bounds[1]:
                    placed = (x, y)
            xs[row], ys[row] = placed
        if key == first_tile and n_rows >= 12:
            # The vendor keeps a few rows just below zero at the section edge.
            ys[-2:] = [-0.125, -0.046875]
        qvs = np.where(
            rng.random(n_rows) < low_qv_fraction,
            rng.choice(qv_low, n_rows),
            rng.choice(qv_high, n_rows),
        )
        codewords = np.array(
            [rng.choice(np.flatnonzero(codeword_gene_mapping == g)) for g in genes],
            dtype=np.int64,
        )
        frame = pd.DataFrame(
            {
                "tile": key,
                "x": xs.astype(np.float32),
                "y": ys.astype(np.float32),
                "z": _snap(rng.uniform(6.9, 31.0, n_rows)).astype(np.float32),
                "gene": genes.astype(np.int64),
                "codeword": codewords,
                "qv": qvs.astype(np.float16),
            }
        )
        frame = frame.sort_values(["gene", "codeword"], kind="stable").reset_index(
            drop=True
        )
        col = np.floor(frame["x"].to_numpy(np.float64) / PIXEL_SIZE_UM).astype(int)
        row_px = np.floor(frame["y"].to_numpy(np.float64) / PIXEL_SIZE_UM).astype(int)
        on_image = (col >= 0) & (col < width) & (row_px >= 0) & (row_px < height)
        label = np.zeros(len(frame), dtype=np.int64)
        label[on_image] = cell_mask[row_px[on_image], col[on_image]]
        on_nucleus = np.zeros(len(frame), dtype=bool)
        on_nucleus[on_image] = nucleus_mask[row_px[on_image], col[on_image]] > 0
        assigned = (label > 0) & (frame["qv"].to_numpy(np.float64) >= 20.0)
        cell_row = np.where(assigned, label - 1, -1)
        frame["prefix"] = np.where(
            assigned, cell_prefix[np.maximum(cell_row, 0)], UNASSIGNED
        ).astype(np.uint32)
        frame["suffix"] = np.where(
            assigned, cell_suffix[np.maximum(cell_row, 0)], UNASSIGNED
        ).astype(np.uint32)
        frame["overlaps_nucleus"] = (assigned & on_nucleus).astype(np.uint8)
        frame["cell_row"] = cell_row
        frame["row"] = np.arange(len(frame), dtype=np.int64)
        frame["transcript_id"] = np.arange(len(frame), dtype=np.uint64) + offset + 1
        offset += len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _create(
    group: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    chunks: tuple[int, ...],
    order: str = "C",
) -> zarr.Array:
    array = group.create_array(
        name,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        order=order,
        compressors=numcodecs.Blosc(
            cname="zstd", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE
        ),
    )
    if data.size:
        array[...] = data
    return array


def _zip_directory(source: Path, destination: Path) -> None:
    """Zip a zarr directory with every member STORED, like the vendor zips."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source).as_posix())


def _new_v2_group(workdir: Path) -> zarr.Group:
    return zarr.open_group(zarr.storage.LocalStore(workdir), mode="w", zarr_format=2)


def _write_transcripts_zip(
    path: Path,
    transcripts: pd.DataFrame,
    *,
    tiles: Mapping[str, int],
    order: Sequence[str],
    gene_names: Sequence[str],
    codeword_gene_mapping: np.ndarray,
    category: np.ndarray,
    category_columns: Sequence[str],
    major_version: int,
) -> None:
    n_genes = len(gene_names)
    with tempfile.TemporaryDirectory() as tmp:
        group = _new_v2_group(Path(tmp))
        _create(
            group,
            "gene_category",
            np.asfortranarray(category),
            chunks=category.shape,
            order="F",
        ).attrs.update({"column_names": list(category_columns)})
        _create(
            group,
            "codeword_category",
            np.asfortranarray(category[codeword_gene_mapping]),
            chunks=(len(codeword_gene_mapping), len(category_columns)),
            order="F",
        ).attrs.update({"column_names": list(category_columns)})
        grid = group.require_group("grid")
        for key in order:
            tile = grid.require_group(key)
            frame = transcripts[transcripts["tile"] == key]
            counts = np.bincount(frame["gene"].to_numpy(), minlength=n_genes)
            ends = np.cumsum(counts)
            gene_offset = np.column_stack([ends - counts, ends]).astype(np.uint32)
            _create(tile, "gene_offset", gene_offset, chunks=gene_offset.shape)
            location = np.asfortranarray(
                frame[["x", "y", "z"]].to_numpy(np.float32).reshape(-1, 3)
            )
            _create(tile, "location", location, chunks=(16, 3), order="F").attrs.update(
                {"column_names": ["x_position", "y_position", "z_position"]}
            )
            _create(
                tile,
                "codeword_identity",
                frame["codeword"].to_numpy(np.uint32).reshape(-1, 1),
                chunks=(16, 1),
            )
            cell_id = np.asfortranarray(
                frame[["prefix", "suffix"]].to_numpy(np.uint32).reshape(-1, 2)
            )
            _create(tile, "cell_id", cell_id, chunks=(16, 2), order="F")
            _create(
                tile,
                "overlaps_nucleus",
                frame["overlaps_nucleus"].to_numpy(np.uint8).reshape(-1, 1),
                chunks=(16, 1),
            )
            _create(
                tile,
                "quality_score",
                frame["qv"].to_numpy(np.float16).reshape(-1, 1),
                chunks=(16, 1),
            )
        gene_counts = np.bincount(transcripts["gene"].to_numpy(), minlength=n_genes)
        codeword_counts = np.bincount(
            transcripts["codeword"].to_numpy(), minlength=len(codeword_gene_mapping)
        )
        grid.attrs.update(
            {
                "grid_key_names": ["grid_x_loc", "grid_y_loc"],
                "grid_size": GRID_SIZE_UM,
                "grid_keys": list(order),
                "grid_number_objects": [int(tiles[key]) for key in order],
                "gene_counts": gene_counts.tolist(),
                "codeword_counts": codeword_counts.tolist(),
            }
        )
        group.attrs.update(
            {
                "name": "RnaDataset",
                "major_version": major_version,
                "minor_version": 0,
                "dataset_uuid": FAKE_IDENTIFIERS["dataset_uuid"],
                "data_format": 0,
                "number_rnas": len(transcripts),
                "spatial_units": "micron",
                "number_genes": n_genes,
                "gene_names": list(gene_names),
                "gene_index_map": {name: i for i, name in enumerate(gene_names)},
                "codeword_count": len(codeword_gene_mapping),
                "codeword_gene_mapping": codeword_gene_mapping.tolist(),
                "codeword_gene_names": [gene_names[g] for g in codeword_gene_mapping],
                "coordinate_space": "refined-final_global_micron",
                "kit_info": [
                    {
                        "codeword_count": len(codeword_gene_mapping),
                        "codeword_indices_start_inclusive": 0,
                        "codeword_indices_end_exclusive": len(codeword_gene_mapping),
                        "codeword_set_name": "Synthetic Whole Transcriptome",
                        "kit_name": "rna_kit_wta",
                    }
                ],
            }
        )
        _zip_directory(Path(tmp), path)


def _rect_polygon(rect: tuple[int, int, int, int]) -> np.ndarray:
    """Vendor polygon of a pixel rectangle: boundary-pixel centres at i * ps."""
    r0, r1, c0, c1 = rect
    corners = [(c0, r0), (c0, r1 - 1), (c1 - 1, r1 - 1), (c1 - 1, r0), (c0, r0)]
    return np.asarray(corners, dtype=np.float64) * PIXEL_SIZE_UM


def _polygon_rows(
    polygons: Sequence[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.zeros((len(polygons), 50), dtype=np.float32)
    num_vertices = np.zeros(len(polygons), dtype=np.int32)
    for row, polygon in enumerate(polygons):
        if polygon is None:
            continue
        padded = np.vstack([polygon, np.repeat(polygon[:1], 25 - len(polygon), axis=0)])
        vertices[row] = padded.astype(np.float32).ravel()
        num_vertices[row] = len(polygon)
    return vertices, num_vertices


def _cell_summary(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    nucleus_cell_index: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """``cell_summary`` in the corner convention: centroid = (mean index + 0.5) ps."""
    summary = np.full((n_cells, 8), np.nan)
    pixel_area = PIXEL_SIZE_UM**2
    for cell in range(n_cells):
        rows, cols = np.nonzero(cell_mask == cell + 1)
        summary[cell, 0] = (cols.mean() + 0.5) * PIXEL_SIZE_UM
        summary[cell, 1] = (rows.mean() + 0.5) * PIXEL_SIZE_UM
        summary[cell, 2] = len(rows) * pixel_area
        owned = np.flatnonzero(nucleus_cell_index == cell)
        summary[cell, 6] = 3.0 if cell == MULTI_COMPONENT_CELL else 0.0
        summary[cell, 7] = len(owned)
        if len(owned):
            sizes = [int((nucleus_mask == n + 1).sum()) for n in owned]
            largest = owned[int(np.argmax(sizes))] + 1
            n_rows, n_cols = np.nonzero(nucleus_mask == largest)
            summary[cell, 3] = (n_cols.mean() + 0.5) * PIXEL_SIZE_UM
            summary[cell, 4] = (n_rows.mean() + 0.5) * PIXEL_SIZE_UM
            summary[cell, 5] = len(n_rows) * pixel_area
    return summary


def _write_cells_zip(
    path: Path,
    *,
    rects: Sequence[tuple[tuple[int, int, int, int], ...]],
    nuclei: Sequence[tuple[tuple[int, int, int, int], ...]],
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    nucleus_cell_index: np.ndarray,
    cell_prefix: np.ndarray,
    cell_suffix: np.ndarray,
    summary: np.ndarray,
    major_version: int,
) -> None:
    n_cells = len(rects)
    with tempfile.TemporaryDirectory() as tmp:
        group = _new_v2_group(Path(tmp))
        _create(
            group, "cell_id", np.column_stack([cell_prefix, cell_suffix]), chunks=(4, 1)
        )
        _create(group, "cell_summary", summary, chunks=(n_cells, 1)).attrs.update(
            {
                "column_names": [
                    "cell_centroid_x",
                    "cell_centroid_y",
                    "cell_area",
                    "nucleus_centroid_x",
                    "nucleus_centroid_y",
                    "nucleus_area",
                    "z_level",
                    "nucleus_count",
                ]
            }
        )
        bboxes = []
        for cell_rects in rects:
            r0 = min(rect[0] for rect in cell_rects)
            r1 = max(rect[1] for rect in cell_rects)
            c0 = min(rect[2] for rect in cell_rects)
            c1 = max(rect[3] for rect in cell_rects)
            bboxes.extend(
                np.asarray([c0, r0, c1, r1], dtype=np.float64) * PIXEL_SIZE_UM
            )
        _create(group, "bboxes", np.asarray(bboxes, dtype=np.float32), chunks=(16,))
        group.require_group("gridded_polygon_sets")
        nucleus_rects = [rect for cell_nuclei in nuclei for rect in cell_nuclei]
        nucleus_polygons: list[np.ndarray | None] = [
            _rect_polygon(rect) for rect in nucleus_rects
        ]
        empty_row = int(np.flatnonzero(nucleus_cell_index == EMPTY_POLYGON_CELL)[0])
        nucleus_polygons[empty_row] = None
        # One vendor polygon per cell: the largest component of a split label.
        cell_polygons: list[np.ndarray | None] = [
            _rect_polygon(max(r, key=lambda q: (q[1] - q[0]) * (q[3] - q[2])))
            for r in rects
        ]
        nucleus_methods = np.where(np.arange(len(nucleus_rects)) % 4 == 3, 4, 3).astype(
            np.uint32
        )
        cell_methods = (np.arange(n_cells) % 3).astype(np.uint32)
        for index, polygons, cell_index, methods in (
            ("0", nucleus_polygons, nucleus_cell_index, nucleus_methods),
            ("1", cell_polygons, np.arange(n_cells, dtype=np.uint32), cell_methods),
        ):
            vertices, num_vertices = _polygon_rows(polygons)
            sub = group.require_group(f"polygon_sets/{index}")
            _create(sub, "cell_index", cell_index.astype(np.uint32), chunks=(8,))
            _create(sub, "method", methods, chunks=(8,))
            _create(sub, "num_vertices", num_vertices, chunks=(8,))
            _create(sub, "vertices", vertices, chunks=(8, 13))
        masks = group.require_group("masks")
        _create(masks, "0", nucleus_mask, chunks=(32, 48))
        _create(masks, "1", cell_mask, chunks=(32, 48))
        transform = np.diag(
            [np.float32(1.0 / PIXEL_SIZE_UM), np.float32(1.0 / PIXEL_SIZE_UM), 1, 1]
        ).astype(np.float32)
        _create(masks, "homogeneous_transform", transform, chunks=(4, 4))
        group.attrs.update(
            {
                "has_gridded_polygon_sets": True,
                "major_version": major_version,
                "minor_version": 3,
                "name": "CellSegmentationDataset",
                "number_cells": n_cells,
                "polygon_set_descriptions": [
                    "DAPI-based nuclei segmentation",
                    "Cell Segmentation",
                ],
                "polygon_set_display_names": ["Nucleus boundaries", "Cell boundaries"],
                "polygon_set_names": ["nucleus", "cell"],
                "segmentation_methods": list(SEGMENTATION_METHODS),
                "spatial_units": "microns",
            }
        )
        _zip_directory(Path(tmp), path)


def _categorical(values: Sequence[object]) -> pd.Categorical:
    return pd.Categorical(values)


def _write_cfm_zip(
    path: Path,
    rng: np.random.Generator,
    *,
    counts: np.ndarray,
    filtered: np.ndarray,
    summary: np.ndarray,
    cell_prefix: np.ndarray,
    cell_suffix: np.ndarray,
    with_is_sketch: bool,
) -> None:
    n_cells = counts.shape[0]
    feature_types = np.asarray([feature[1] for feature in FEATURES])
    is_gene_type = feature_types == "Gene Expression"
    packed = (cell_suffix.astype(np.int64) << 32) | cell_prefix.astype(np.int64)

    def masked(values: Sequence[object]) -> list[object]:
        return [
            value if keep else np.nan
            for value, keep in zip(values, filtered, strict=True)
        ]

    def per_type(feature_type: str) -> np.ndarray:
        return counts[:, feature_types == feature_type].sum(axis=1)

    def cycled(values: Sequence[str]) -> pd.Categorical:
        return _categorical(masked((list(values) * n_cells)[:n_cells]))

    obs = pd.DataFrame(
        {
            "cell_id": packed,
            "filtered": filtered,
            "leiden_res_1.0": cycled(["0", "1"]),
            "centroid_column": summary[:, 0].astype(np.float32),
            "centroid_row": summary[:, 1].astype(np.float32),
            "cell_area": summary[:, 2].astype(np.float32),
            "nucleus_area": summary[:, 5].astype(np.float32),
            "transcript_counts": counts[:, is_gene_type].sum(axis=1),
            "control_probe_counts": per_type("Negative Control Probe"),
            "control_codeword_counts": per_type("Negative Control Codeword"),
            "total_counts": counts.sum(axis=1),
            "azimuth_broad": cycled(["Neuron", "Glial cell"]),
            "azimuth_coarse": cycled(["Excitatory", "Unassigned"]),
            "azimuth_fine": cycled(["Fine A", "Fine Not Confidently Annotated"]),
            "azimuth_score": np.where(filtered, 0.9, np.nan).astype(np.float32),
            "azimuth_full_hierarchy": cycled(["Neuron|Excitatory"]),
        },
        index=pd.Index(cell_id_strings(cell_prefix, cell_suffix), name="barcode"),
    )
    if with_is_sketch:
        obs["is_sketch"] = False
    var = pd.DataFrame(
        {
            "feature_name": [feature[0] for feature in FEATURES],
            "filtered": is_gene_type & (np.arange(len(FEATURES)) != 3),
            "highly_variable": np.arange(len(FEATURES)) % 2 == 0,
            "feature_type": _categorical(feature_types),
            "genome": _categorical(["GRCh38"] * len(FEATURES)),
            "de_leiden_res_1.0_c0_logfc": rng.normal(size=len(FEATURES)),
        },
        index=pd.Index([feature[3] for feature in FEATURES], name="feature_id"),
    )
    adata = ad.AnnData(X=sparse.csr_matrix(counts.astype(np.int32)), obs=obs, var=var)
    pca = rng.normal(size=(n_cells, 10)).astype(np.float32)
    pca[~filtered] = np.nan
    adata.obsm["X_pca"] = pca
    adata.obsm["X_umap"] = pca[:, :2].copy()
    with tempfile.TemporaryDirectory() as tmp:
        group = _new_v2_group(Path(tmp))
        ad.io.write_elem(group, "/", adata)
        _zip_directory(Path(tmp), path)


def _ome_xml(
    channels: Sequence[tuple[str, str]],
    files: Sequence[str],
    uuids: Sequence[str],
    own: int,
    shape: tuple[int, int],
) -> bytes:
    channel_xml = "".join(
        f'<Channel ID="Channel:{k}" Name="{name}" SamplesPerPixel="1"/>'
        for k, (name, _) in enumerate(channels)
    )
    tiff_data = "".join(
        f'<TiffData FirstC="{k}" FirstT="0" FirstZ="0" PlaneCount="1">'
        f'<UUID FileName="{name}">{uid}</UUID></TiffData>'
        for k, (name, uid) in enumerate(zip(files, uuids, strict=True))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
        f'UUID="{uuids[own]}"><Image ID="Image:0" Name="Baseline">'
        '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" '
        f'SizeX="{shape[1]}" SizeY="{shape[0]}" SizeZ="1" SizeC="{len(channels)}" '
        f'SizeT="1" PhysicalSizeX="{PIXEL_SIZE_UM}" PhysicalSizeXUnit="µm" '
        f'PhysicalSizeY="{PIXEL_SIZE_UM}" PhysicalSizeYUnit="µm">'
        f"{channel_xml}{tiff_data}</Pixels></Image></OME>"
    ).encode()


def write_ome_channel(
    path: Path, level0: np.ndarray, description: bytes, n_levels: int
) -> None:
    """Write one channel as a tiled JPEG-2000 OME-TIFF with SubIFD levels.

    Levels halve with floor rounding (2 x 2 means), like the vendor pyramid.

    Args:
        path: Output file.
        level0: Full-resolution uint16 plane.
        description: OME-XML of the multi-file series.
        n_levels: Number of stored levels, level 0 included.
    """
    levels = [level0]
    while len(levels) < n_levels:
        previous = levels[-1]
        height, width = previous.shape[0] // 2, previous.shape[1] // 2
        block = previous[: 2 * height, : 2 * width].astype(np.float64)
        levels.append(
            block.reshape(height, 2, width, 2).mean(axis=(1, 3)).astype(np.uint16)
        )
    options = {
        "tile": TIFF_TILE,
        "compression": "jpeg2000",
        "compressionargs": {"level": 0},
        "photometric": "minisblack",
        "metadata": None,
    }
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        writer.write(
            levels[0], subifds=len(levels) - 1, description=description, **options
        )
        for level in levels[1:]:
            writer.write(level, subfiletype=1, **options)


def _write_morphology(
    directory: Path,
    rng: np.random.Generator,
    *,
    channels: Sequence[tuple[str, str]],
    shape: tuple[int, int],
    file_order_differs: bool,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    directory.mkdir(parents=True, exist_ok=True)
    n = len(channels)
    positions = [n - 1 - k for k in range(n)] if file_order_differs else list(range(n))
    files = [
        f"ch{positions[k]:04d}_{slug}.ome.tif" for k, (_, slug) in enumerate(channels)
    ]
    uuids = [f"urn:uuid:{uuid.UUID(int=0xA7E4A000 + k)}" for k in range(n)]
    planes = []
    for k in range(n):
        plane = (rng.integers(0, 500, shape) + 1000 * (k + 1)).astype(np.uint16)
        planes.append(plane)
        write_ome_channel(
            directory / files[k],
            plane,
            _ome_xml(channels, files, uuids, k, shape),
            N_STORED_LEVELS,
        )
    return tuple(files), tuple(planes)


def _write_small_files(
    root: Path,
    *,
    organism: str,
    analysis_sw_version: str,
    n_cells: int,
    n_transcripts: int,
    four_channels: bool,
) -> None:
    analysis_summary = f"analysis_summary_{RUN_SUFFIX}.html"
    experiment = {
        "major_version": 1,
        "minor_version": 2,
        "run_name": FAKE_IDENTIFIERS["run_name"],
        "run_start_time": "2000-01-01T00:00:00Z",
        "slide_name": FAKE_IDENTIFIERS["slide_name"],
        "slide_id": FAKE_IDENTIFIERS["slide_id"],
        "region_name": FAKE_IDENTIFIERS["region_name"],
        "slide_type": "Synthetic glass",
        "sample_type": "Fresh frozen",
        "panel_config": {"panel_config_name": "", "panel_config_id": "hWTA_v1"},
        "panel_wta": {
            "panel_name": "Synthetic Whole Transcriptome",
            "panel_design_id": "hWTA_v1",
            "panel_num_targets": 7,
            "panel_organism": organism,
            "panel_tissue_type": "Multi",
        },
        "total_num_targets": 7,
        "segmentation_stain": "atera_cell_segmentation_stains_v1"
        if four_channels
        else "",
        "chemistry_version": "Atera v1",
        "pixel_size": PIXEL_SIZE_UM,
        "z_step_size": 3.0,
        "instrument_sn": FAKE_IDENTIFIERS["instrument_sn"],
        "instrument_sw_version": "1.0.0.79",
        "analysis_sw_version": analysis_sw_version,
        "analysis_uuid": FAKE_IDENTIFIERS["analysis_uuid"],
        "experiment_uuid": FAKE_IDENTIFIERS["experiment_uuid"],
        "cassette_uuid": FAKE_IDENTIFIERS["cassette_uuid"],
        "roi_uuid": FAKE_IDENTIFIERS["roi_uuid"],
        "well_uuid": FAKE_IDENTIFIERS["well_uuid"],
        "calibration_uuid": FAKE_IDENTIFIERS["calibration_uuid"],
        "num_cells": n_cells,
        "num_transcripts": n_transcripts,
        "num_transcripts_high_quality": n_transcripts,
        "fraction_transcripts_assigned": 0.5,
        "imported_cell_frac": "NaN",
        "images": {
            "morphology_2d_filepath": "morphology_2d/ch0000_dapi.ome.tif",
            "morphology_3d_filepath": "morphology_3d/ch0000_dapi_3d.ome.tif",
        },
        "explorer_files": {
            "transcripts_zarr_filepath": "transcripts.zarr.zip",
            "transcripts_viz_zarr_filepath": "binned_transcripts.zarr.zip",
            "cells_zarr_filepath": "cells.zarr.zip",
            "cell_feature_zarr_filepath": "cell_feature_matrix.zarr.zip",
            "cell_feature_viz_zarr_filepath": "csc_cell_feature_matrix.zarr.zip",
            "analysis_summary_filepath": analysis_summary,
        },
    }
    (root / "experiment.spatial").write_text(json.dumps(experiment, indent=2))
    targets = []
    for symbol, feature_type, _, feature_id, codewords in FEATURES:
        if symbol == "BOOSTGENE1" or feature_type in (
            "Negative Control Codeword",
            "Unassigned Codeword",
        ):
            continue
        descriptor = {
            "Gene Expression": "gene",
            "Negative Control Probe": "negative_control",
            "Genomic Control": "genomic_control",
            "Deprecated Codeword": "deprecated_codeword",
        }[feature_type]
        targets.append(
            {
                "codewords": list(range(codewords)),
                "info": {"gene_coverage": 3 + len(targets), "reference": "hg38"},
                "source": {"category": "current"},
                "type": {
                    "data": {"id": feature_id, "name": symbol},
                    "descriptor": descriptor,
                },
            }
        )
    panel = {
        "spec_version": "1.1.1",
        "metadata": {"ui_props": {"visible": True, "display_name": ""}},
        "payloads": [
            {
                "chemistry": {"decoder": "atera_decoding_v1", "version": "v3"},
                "customer": {
                    "email": FAKE_IDENTIFIERS["customer_email"],
                    "name": FAKE_IDENTIFIERS["customer_name"],
                },
                "designer": {
                    "date_created": "2000-01-01T00:00:00",
                    "designed_by": {
                        "email": FAKE_IDENTIFIERS["designer_email"],
                        "name": FAKE_IDENTIFIERS["designer_name"],
                    },
                    "version": "1",
                },
                "panel": {
                    "identity": {
                        "design_id": "hWTA_v1",
                        "name": "Synthetic WTA",
                        "version": "1.0.0",
                    },
                    "num_gene_targets": 7,
                    "species": organism,
                    "type": {"descriptor": "predesigned"},
                },
                "targets": targets,
            }
        ],
    }
    (root / "panel_config.json").write_text(json.dumps(panel))
    header = (
        "run_name,instrument_sn,slide_name,region_name,slide_type,sample_type,slide_id,"
        "run_start_time,panel_config_id,panel_wta_name,panel_wta_design_id,"
        "reagent_plate_1,reagent_pouch_1,num_cells_detected,"
        "fraction_transcripts_assigned,median_genes_per_cell_panel_a"
    )
    values = (
        f"{FAKE_IDENTIFIERS['run_name']},{FAKE_IDENTIFIERS['instrument_sn']},"
        f"{FAKE_IDENTIFIERS['slide_name']},{FAKE_IDENTIFIERS['region_name']},"
        f"Synthetic glass,Fresh frozen,{FAKE_IDENTIFIERS['slide_id']},"
        f"{FAKE_IDENTIFIERS['run_start_time']},hWTA_cfg,"
        f"{FAKE_IDENTIFIERS['panel_label']},hWTA_v1,{FAKE_IDENTIFIERS['reagent_lot']},"
        f"424242,{n_cells},0.5,"
    )
    (root / f"metrics_summary_{RUN_SUFFIX}.csv").write_text(f"{header}\n{values}\n")
    (root / analysis_summary).write_text(
        f"<html><body>{FAKE_IDENTIFIERS['run_name']}</body></html>"
    )
    fovs = {
        "units": "micron",
        "fov_locations": {
            "AA1": {"x": -0.5, "y": -0.5, "width": 11.0, "height": 11.0},
            "AA2": {"x": 10.0, "y": -0.5, "width": 11.0, "height": 11.0},
        },
    }
    (root / "morphology_fov_locations.json").write_text(json.dumps(fovs))
    overview_fovs = {
        "units": "pixel",
        "fov_locations": {"AA1": {"x": 3, "y": 4, "width": 50, "height": 50}},
    }
    (root / "overview_scan_fov_locations.json").write_text(json.dumps(overview_fovs))
    # Copied verbatim, never decoded: the PNG signature plus fake bytes is enough.
    (root / "overview_scan.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)))
    (root / "qc_images.html").write_text(
        f"<html><body>{FAKE_IDENTIFIERS['slide_name']}</body></html>"
    )
    (root / f"{COMPANION_PREFIX}_he_imagealignment.csv").write_text(
        "1.0,0.0,2.5\n0.0,1.0,-1.5\n0.0,0.0,1.0\n"
    )
    (root / f"{COMPANION_PREFIX}_keypoints.csv").write_text(
        "fixedX,fixedY,alignmentX,alignmentY\n1.0,2.0,3.5,0.5\n"
    )
    diffexp = root / "diffexp_genes"
    diffexp.mkdir(exist_ok=True)
    (diffexp / "gene_expression_graphclust.csv").write_text(
        "Feature ID,Feature Name,Cluster 0 Mean Counts\nENSG00000132639,SNAP25,1.0\n"
    )


def rewrite_zip_store(zip_path: Path, edit: Callable[[zarr.Group], None]) -> None:
    """Rewrite a zipped zarr v2 store in place after ``edit`` changes its group.

    For failure tests: ``edit`` receives the writable root group of an
    extracted copy (change attrs or array values), which is then zipped back.

    Args:
        zip_path: A zipped store of a written bundle.
        edit: Function that changes the root group.
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "store"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(workdir)
        edit(zarr.open_group(zarr.storage.LocalStore(workdir), mode="r+"))
        _zip_directory(workdir, zip_path)
