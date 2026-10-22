"""Veto of Cellpose cell masks that sit on a modelled coverslip bead.

MERSCOPE bead removal subtracts a model of every fiducial bead from the stains
before the max projection (see :mod:`merxen.io.bead_removal`). What the model
leaves behind (and what the fill does not catch) can still be segmented by
Cellpose as a "cell": a ring or disc centred on the bead, with no nucleus. This
module drops such masks, after Cellpose has produced the stitched and filtered cell
masks and before they are saved and the transcripts are seeded for ProSeg.

A mask is dropped when both hold:

* its centroid lies within ``max(min_distance_px, radius_fraction x r)`` of a
  modelled bead with ``A0 >= a0_min`` (``r = sqrt(area / pi)``), pruned and blob
  beads included, and
* it has no DAPI-positive nucleus: the ``dapi_percentile`` (90th) of the DAPI inside
  the mask does not exceed the median of the DAPI in a ring 10 to 25 px around the
  mask plus ``dapi_noise_multiple`` (3) robust noise units (1.4826 x the median
  absolute deviation of that ring), and it does not reach the tissue DAPI level of
  the slide either (``tissue_floor``; the Otsu threshold of the log DAPI of the
  coarsest projection level). In tissue the ring takes in neighbouring nuclei, so
  the ring test alone drops real cells that happen to lie on a bead; the floor keeps
  them, while bead "cells" on the glass stay far below it. The DAPI is the DAPI
  channel of the same projection Cellpose segmented. The separate nuclei
  segmentation is deliberately not used (it runs in its own task and misses
  nuclei).

The bead table is ``bead_removal_qc/beads.csv.gz`` of the source SpatialData zarr
(coordinates in mosaic pixels, the pixel grid of the masks). The step does nothing
when the source was built without bead removal or has no bead table (the masks stay
bit-identical and no file is written); otherwise it records ``bead_mask_veto.json``
(parameters and counts) next to the masks, and the parameters go into the mask
sidecar of :mod:`merxen.io.bead_removal.provenance`, so masks computed with the
veto are never reused for a request without it, or the reverse.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from merxen.config import BeadMaskVetoConfig
from merxen.io.bead_removal.provenance import segmentation_identity
from merxen.io.image_source import (
    MERSCOPE_ZPROJ_IMAGE_NAME,
    build_image_source,
    fetch_tile,
    image_to_cyx,
)
from merxen.segmentation.mask_filter import drop_labels_from_mask

logger = logging.getLogger(__name__)

# The QC folder the build writes into the source zarr (``qc.QC_DIRNAME``; repeated
# here so that this module does not import the QC code, a test keeps them equal).
BEAD_QC_DIRNAME = "bead_removal_qc"
BEAD_TABLE_NAME = "beads.csv.gz"
VETO_SIDECAR_NAME = "bead_mask_veto.json"
VETO_SIDECAR_SCHEMA = 1
# Bump when the decision rule changes: masks of another version are not reused.
# 2: the tissue DAPI floor (``tissue_floor``).
VETO_ALGORITHM_VERSION = 2
DAPI_CHANNEL = "DAPI"
# Robust sd of a Gaussian from the median absolute deviation.
MAD_TO_SD = 1.4826
# Largest side (px) of the overview the tissue DAPI floor is computed on; a coarsest
# projection level larger than this is subsampled.
FLOOR_OVERVIEW_MAX_PX = 4096

DapiReader = Callable[[int, int, int, int], np.ndarray]


@dataclass(frozen=True)
class PreparedBeadMaskVeto:
    """Everything the veto needs, resolved before Cellpose starts.

    Attributes:
        config: Method parameters.
        identity: The parameters that decide the masks (``bead_mask_veto_identity``).
        bead_y: Row (mosaic px) of every bead with ``A0 >= config.a0_min``.
        bead_x: Column (mosaic px) of the same beads.
        n_table_beads: Number of beads in the table, before the ``A0`` limit.
        read_dapi: ``read_dapi(y0, y1, x0, x1)`` returns the DAPI of that window
            of the projection Cellpose segmented, float32 ``[y1 - y0, x1 - x0]``.
        dapi_floor: Tissue DAPI level of the slide; a mask whose DAPI percentile
            reaches it is kept. ``None`` when ``tissue_floor`` is off or the level
            cannot be estimated.
    """

    config: BeadMaskVetoConfig
    identity: dict[str, Any]
    bead_y: np.ndarray
    bead_x: np.ndarray
    n_table_beads: int
    read_dapi: DapiReader
    dapi_floor: float | None = None


@dataclass(frozen=True)
class BeadMaskVetoResult:
    """Counts of one veto run.

    Attributes:
        n_masks: Cell masks before the veto.
        n_masks_after: Cell masks after the veto.
        n_beads_used: Beads with ``A0 >= a0_min`` that were tested against.
        n_table_beads: Beads in the bead table.
        n_near_bead: Masks whose centroid is near a used bead.
        n_vetoed: Masks dropped (near a bead and without a DAPI-positive nucleus).
        n_kept_with_nucleus: Masks near a bead that were kept for their nucleus.
        n_kept_unmeasurable: Masks near a bead that were kept because the DAPI
            background ring had too few pixels.
        written: Whether the mask file was rewritten (False when nothing dropped).
        dapi_floor: The tissue DAPI floor that was applied, or ``None``.
    """

    n_masks: int
    n_masks_after: int
    n_beads_used: int
    n_table_beads: int
    n_near_bead: int
    n_vetoed: int
    n_kept_with_nucleus: int
    n_kept_unmeasurable: int
    written: bool
    dapi_floor: float | None = None

    def counts(self) -> dict[str, int | bool | float | None]:
        """Return the counts (and the applied DAPI floor) as a JSON-safe mapping."""
        return {
            "n_masks": self.n_masks,
            "n_masks_after": self.n_masks_after,
            "n_beads_used": self.n_beads_used,
            "n_table_beads": self.n_table_beads,
            "n_near_bead": self.n_near_bead,
            "n_vetoed": self.n_vetoed,
            "n_kept_with_nucleus": self.n_kept_with_nucleus,
            "n_kept_unmeasurable": self.n_kept_unmeasurable,
            "written": self.written,
            "dapi_floor": self.dapi_floor,
        }


def bead_table_path(source: Path | str) -> Path:
    """Return the bead table of a source SpatialData zarr.

    Args:
        source: The source SpatialData zarr (a symlink is followed).

    Returns:
        Path of ``bead_removal_qc/beads.csv.gz`` inside it (it may not exist).
    """
    return Path(source) / BEAD_QC_DIRNAME / BEAD_TABLE_NAME


def veto_sidecar_path(mask_path: Path | str) -> Path:
    """Return the veto record that sits next to a Cellpose cell mask."""
    return Path(mask_path).with_name(VETO_SIDECAR_NAME)


def remove_veto_sidecar(mask_path: Path | str) -> None:
    """Delete the veto record of a mask that is about to be regenerated."""
    veto_sidecar_path(mask_path).unlink(missing_ok=True)


def bead_mask_veto_identity(
    config: BeadMaskVetoConfig,
    source: Path | str,
    recorded: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the parameters of the veto that a run on this source applies.

    This is what the mask sidecar records and what a reuse check compares.

    Args:
        config: Veto configuration.
        source: The source SpatialData zarr.
        recorded: ``merxen_image_preprocessing`` of the source, or ``None`` for a
            legacy store.

    Returns:
        ``None`` when the veto does nothing: it is disabled, the source was built
        without bead removal, or it has no bead table. Otherwise the decision
        parameters (not the chunk size) and the algorithm version.
    """
    if (
        not config.enabled
        or segmentation_identity(recorded) is None
        or not bead_table_path(source).is_file()
    ):
        return None
    return {
        "version": VETO_ALGORITHM_VERSION,
        "a0_min": float(config.a0_min),
        "min_distance_px": float(config.min_distance_px),
        "radius_fraction": float(config.radius_fraction),
        "ring_px": [float(config.ring_inner_px), float(config.ring_outer_px)],
        "dapi_percentile": float(config.dapi_percentile),
        "dapi_noise_multiple": float(config.dapi_noise_multiple),
        "min_ring_px": int(config.min_ring_px),
        "tissue_floor": bool(config.tissue_floor),
    }


def read_bead_table(path: Path | str) -> pd.DataFrame:
    """Read the bead table written by the bead-removal QC.

    Args:
        path: ``beads.csv.gz``.

    Returns:
        Data frame with at least the columns ``y``, ``x`` (mosaic px) and ``A0``;
        rows with a missing coordinate or amplitude are dropped. Pruned and blob
        beads are kept.

    Raises:
        ValueError: A required column is missing.
    """
    table = pd.read_csv(path)
    missing = {"y", "x", "A0"} - set(table.columns)
    if missing:
        raise ValueError(
            f"{path} lacks the column(s) {sorted(missing)}; expected the bead table "
            "of the bead-removal QC"
        )
    return table.dropna(subset=["y", "x", "A0"]).reset_index(drop=True)


def make_dapi_reader(sdata: Any, shape: tuple[int, int]) -> DapiReader:
    """Build a reader of the DAPI channel of the projection Cellpose segments.

    Args:
        sdata: The source SpatialData (read from the zarr).
        shape: ``(height, width)`` of the Cellpose masks.

    Returns:
        ``read_dapi(y0, y1, x0, x1)`` returning the float32 DAPI window.

    Raises:
        ValueError: The MERSCOPE projection image, its DAPI channel, or a matching
            image size is missing.
    """
    images = getattr(sdata, "images", None)
    if images is None or MERSCOPE_ZPROJ_IMAGE_NAME not in images:
        raise ValueError(
            f"the bead-mask veto needs the {MERSCOPE_ZPROJ_IMAGE_NAME} image of the "
            "source SpatialData; set bead_mask_veto.enabled to false "
            "(merscope_bead_mask_veto=false) to segment without it"
        )
    source = build_image_source(
        images[MERSCOPE_ZPROJ_IMAGE_NAME],
        requested_channels=[DAPI_CHANNEL],
        as_float32=True,
    )
    if list(source["channels"]) != [DAPI_CHANNEL]:
        raise ValueError(
            f"the bead-mask veto needs a {DAPI_CHANNEL} channel in "
            f"{MERSCOPE_ZPROJ_IMAGE_NAME}; found {list(source['channels'])}"
        )
    if tuple(source["shape"][:2]) != tuple(shape):
        raise ValueError(
            f"the {MERSCOPE_ZPROJ_IMAGE_NAME} image is {tuple(source['shape'][:2])} "
            f"px but the Cellpose masks are {tuple(shape)} px"
        )

    def read_dapi(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        window = fetch_tile(source, y0, y1, x0, x1)
        return np.asarray(window[..., 0], dtype=np.float32)

    return read_dapi


def tissue_dapi_floor(sdata: Any) -> float | None:
    """Estimate the tissue DAPI level of the projection Cellpose segments.

    The Otsu threshold of ``log1p`` of the positive DAPI of the coarsest level of
    the projection (subsampled to at most ``FLOOR_OVERVIEW_MAX_PX`` per side), back
    in counts. It separates the glass from the tissue, so a DAPI nucleus, or tissue,
    reaches it and a bead "cell" on the glass does not.

    Args:
        sdata: The source SpatialData (read from the zarr).

    Returns:
        The DAPI level in counts, or ``None`` when the overview has fewer than two
        distinct positive values (no threshold).

    Raises:
        ValueError: The projection image or its DAPI channel is missing.
    """
    images = getattr(sdata, "images", None)
    if images is None or MERSCOPE_ZPROJ_IMAGE_NAME not in images:
        raise ValueError(
            f"the bead-mask veto needs the {MERSCOPE_ZPROJ_IMAGE_NAME} image of the "
            "source SpatialData"
        )
    image = images[MERSCOPE_ZPROJ_IMAGE_NAME]
    children = getattr(image, "children", None)
    if children:
        # Multiscale: scale0 ... scaleN, the last the coarsest.
        coarsest = max(
            children, key=lambda name: int("".join(filter(str.isdigit, name)) or 0)
        )
        image = image[coarsest].ds["image"]
    data = image_to_cyx(image)
    channels = [str(c) for c in data.coords["c"].values] if "c" in data.coords else []
    if DAPI_CHANNEL not in channels:
        raise ValueError(
            f"the bead-mask veto needs a {DAPI_CHANNEL} channel in "
            f"{MERSCOPE_ZPROJ_IMAGE_NAME}; found {channels}"
        )
    dapi = data.isel(c=channels.index(DAPI_CHANNEL))
    longest = max(int(dapi.shape[0]), int(dapi.shape[1]))
    step = max(1, -(-longest // FLOOR_OVERVIEW_MAX_PX))
    values = np.asarray(dapi[::step, ::step], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2 or float(values.min()) == float(values.max()):
        return None
    return float(np.expm1(threshold_otsu(np.log1p(values))))


def prepare_bead_mask_veto(
    config: BeadMaskVetoConfig,
    *,
    source: Path | str,
    recorded: Mapping[str, Any] | None,
    sdata: Any,
    shape: tuple[int, int],
) -> PreparedBeadMaskVeto | None:
    """Resolve the veto before Cellpose runs, so that a problem fails early.

    Args:
        config: Veto configuration.
        source: The source SpatialData zarr.
        recorded: ``merxen_image_preprocessing`` of the source, or ``None``.
        sdata: The source SpatialData (read from the zarr).
        shape: ``(height, width)`` of the Cellpose masks.

    Returns:
        ``None`` when the veto does nothing (see :func:`bead_mask_veto_identity`);
        otherwise the prepared veto.

    Raises:
        ValueError: The bead table has no usable columns, or the DAPI projection
            cannot be read (see :func:`make_dapi_reader`).
    """
    identity = bead_mask_veto_identity(config, source, recorded)
    if identity is None:
        return None
    table = read_bead_table(bead_table_path(source))
    used = table[table["A0"] >= config.a0_min]
    return PreparedBeadMaskVeto(
        config=config,
        identity=identity,
        bead_y=used["y"].to_numpy(dtype=np.float64),
        bead_x=used["x"].to_numpy(dtype=np.float64),
        n_table_beads=int(len(table)),
        read_dapi=make_dapi_reader(sdata, shape),
        dapi_floor=tissue_dapi_floor(sdata) if config.tissue_floor else None,
    )


def _rows_per_chunk(shape: tuple[int, ...], dtype: np.dtype[Any], chunk_mb: int) -> int:
    row_bytes = max(1, int(shape[1]) * int(np.dtype(dtype).itemsize))
    return max(1, min(int(shape[0]), (chunk_mb * 1024 * 1024) // row_bytes))


def _label_statistics(
    mask: np.ndarray, chunk_mb: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area and the sums of the row and column of every label, in row chunks.

    Args:
        mask: Two-dimensional non-negative integer label mask (may be a memory map).
        chunk_mb: Memory (MB) of one row chunk.

    Returns:
        ``(area, sum_y, sum_x)``, each of length ``max label + 1``.
    """
    n_labels = int(mask.max()) + 1
    area = np.zeros(n_labels, np.int64)
    sum_y = np.zeros(n_labels, np.float64)
    sum_x = np.zeros(n_labels, np.float64)
    rows = _rows_per_chunk(mask.shape, mask.dtype, chunk_mb)
    for y0 in range(0, int(mask.shape[0]), rows):
        chunk = np.asarray(mask[y0 : y0 + rows])
        ys, xs = np.nonzero(chunk)
        if ys.size == 0:
            continue
        labels = chunk[ys, xs].astype(np.int64, copy=False)
        counted = np.bincount(labels)
        area[: counted.size] += counted
        sum_y[: counted.size] += np.bincount(labels, weights=ys + y0)
        sum_x[: counted.size] += np.bincount(labels, weights=xs)
    return area, sum_y, sum_x


def _candidate_pixels(
    mask: np.ndarray, is_candidate: np.ndarray, chunk_mb: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect the pixels of the candidate labels, in row chunks.

    Args:
        mask: Two-dimensional label mask (may be a memory map).
        is_candidate: Bool lookup, ``is_candidate[label]``.
        chunk_mb: Memory (MB) of one row chunk.

    Returns:
        ``(labels, ys, xs)`` of those pixels, sorted by label.
    """
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rows = _rows_per_chunk(mask.shape, mask.dtype, chunk_mb)
    for y0 in range(0, int(mask.shape[0]), rows):
        chunk = np.asarray(mask[y0 : y0 + rows])
        ys, xs = np.nonzero(is_candidate[chunk])
        if ys.size:
            parts.append((chunk[ys, xs].astype(np.int64), ys + y0, xs))
    if not parts:
        empty = np.empty(0, np.int64)
        return empty, empty, empty
    labels = np.concatenate([p[0] for p in parts])
    ys = np.concatenate([p[1] for p in parts])
    xs = np.concatenate([p[2] for p in parts])
    order = np.argsort(labels, kind="stable")
    return labels[order], ys[order], xs[order]


def has_dapi_nucleus(
    ys: np.ndarray,
    xs: np.ndarray,
    read_dapi: DapiReader,
    shape: tuple[int, int],
    config: BeadMaskVetoConfig,
    dapi_floor: float | None = None,
) -> bool | None:
    """Decide whether one mask contains a DAPI-positive nucleus.

    The ``dapi_percentile`` of the DAPI inside the mask is compared with the median
    of the DAPI in the ring between ``ring_inner_px`` and ``ring_outer_px`` around
    the mask, plus ``dapi_noise_multiple`` robust noise units of that ring. A mask
    whose percentile reaches ``dapi_floor`` has a nucleus whatever the ring says.

    Args:
        ys: Rows (mosaic px) of the mask pixels.
        xs: Columns of the mask pixels.
        read_dapi: Reader of the DAPI of the segmented projection.
        shape: ``(height, width)`` of the mask grid.
        config: Veto parameters.
        dapi_floor: Tissue DAPI level (see :func:`tissue_dapi_floor`), or ``None``
            for the ring test alone.

    Returns:
        True or False, or ``None`` when the ring has fewer than ``min_ring_px``
        finite pixels (the background cannot be estimated).
    """
    pad = int(np.ceil(config.ring_outer_px)) + 1
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, shape[1])
    dapi = read_dapi(y0, y1, x0, x1)
    inside = np.zeros(dapi.shape, bool)
    inside[ys - y0, xs - x0] = True
    mask_values = dapi[inside & np.isfinite(dapi)]
    if mask_values.size == 0:
        return None
    level = float(np.percentile(mask_values, config.dapi_percentile))
    if dapi_floor is not None and level >= dapi_floor:
        return True
    distance = ndi.distance_transform_edt(~inside)
    ring = (
        (distance >= config.ring_inner_px)
        & (distance <= config.ring_outer_px)
        & np.isfinite(dapi)
    )
    if int(ring.sum()) < config.min_ring_px:
        return None
    ring_values = dapi[ring].astype(np.float64)
    background = float(np.median(ring_values))
    noise = MAD_TO_SD * float(np.median(np.abs(ring_values - background)))
    return level > background + config.dapi_noise_multiple * noise


def apply_bead_mask_veto(
    mask_path: Path | str,
    veto: PreparedBeadMaskVeto,
    *,
    dataset_name: str = "",
    show_progress: bool = False,
) -> BeadMaskVetoResult:
    """Drop the cell masks that sit on a modelled bead and have no nucleus.

    The mask file is rewritten (kept labels compactly relabeled, as the area filter
    does) only when a mask is dropped; ``bead_mask_veto.json`` is written next to it
    in every case, and an INFO line reports the counts.

    Args:
        mask_path: Cellpose cell label mask (``.npy``), already stitched and area
            filtered.
        veto: The prepared veto.
        dataset_name: Dataset label for the log line.
        show_progress: Show a progress bar while the mask is rewritten.

    Returns:
        The counts of the run.
    """
    mask_path = Path(mask_path)
    config = veto.config
    mask = np.load(mask_path, mmap_mode="r")
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    area, sum_y, sum_x = _label_statistics(mask, config.chunk_mb)
    present = np.flatnonzero(area > 0)
    near = np.empty(0, np.int64)
    if present.size and veto.bead_y.size:
        centre_y = sum_y[present] / area[present]
        centre_x = sum_x[present] / area[present]
        radius = np.sqrt(area[present] / np.pi)
        limit = np.maximum(config.min_distance_px, config.radius_fraction * radius)
        tree = cKDTree(np.column_stack([veto.bead_y, veto.bead_x]))
        nearest, _ = tree.query(np.column_stack([centre_y, centre_x]))
        near = present[nearest <= limit]

    vetoed: list[int] = []
    n_with_nucleus = 0
    n_unmeasurable = 0
    if near.size:
        is_candidate = np.zeros(area.size, bool)
        is_candidate[near] = True
        labels, ys, xs = _candidate_pixels(mask, is_candidate, config.chunk_mb)
        first = np.searchsorted(labels, near)
        last = np.searchsorted(labels, near, side="right")
        for label, start, stop in zip(near, first, last, strict=True):
            verdict = has_dapi_nucleus(
                ys[start:stop],
                xs[start:stop],
                veto.read_dapi,
                shape,
                config,
                veto.dapi_floor,
            )
            if verdict is None:
                n_unmeasurable += 1
            elif verdict:
                n_with_nucleus += 1
            else:
                vetoed.append(int(label))
    del mask

    written = False
    if vetoed:
        summary = drop_labels_from_mask(
            mask_path,
            vetoed,
            chunk_mb=config.chunk_mb,
            show_progress=show_progress,
        )
        written = bool(summary["written"])
    result = BeadMaskVetoResult(
        n_masks=int(present.size),
        n_masks_after=int(present.size) - len(vetoed),
        n_beads_used=int(veto.bead_y.size),
        n_table_beads=veto.n_table_beads,
        n_near_bead=int(near.size),
        n_vetoed=len(vetoed),
        n_kept_with_nucleus=n_with_nucleus,
        n_kept_unmeasurable=n_unmeasurable,
        written=written,
        dapi_floor=veto.dapi_floor,
    )
    _write_veto_sidecar(mask_path, veto.identity, result)
    prefix = f"[{dataset_name}] " if dataset_name else ""
    logger.info(
        "%sBead mask veto: dropped %d of %d Cellpose masks (%d centred on a modelled "
        "bead with A0 >= %g, %d of %d table beads used; %d kept with a DAPI nucleus, "
        "%d kept without a measurable DAPI background; tissue DAPI floor %s)",
        prefix,
        result.n_vetoed,
        result.n_masks,
        result.n_near_bead,
        config.a0_min,
        result.n_beads_used,
        result.n_table_beads,
        result.n_kept_with_nucleus,
        result.n_kept_unmeasurable,
        "off" if veto.dapi_floor is None else f"{veto.dapi_floor:.1f}",
    )
    return result


def _write_veto_sidecar(
    mask_path: Path, identity: Mapping[str, Any], result: BeadMaskVetoResult
) -> Path:
    """Write ``bead_mask_veto.json`` next to the mask (parameters and counts)."""
    path = veto_sidecar_path(mask_path)
    payload = {
        "schema": VETO_SIDECAR_SCHEMA,
        "parameters": dict(identity),
        "counts": result.counts(),
    }
    # Written through a temporary name so a reader never sees half a file.
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path
