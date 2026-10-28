#!/usr/bin/env python
"""Compare two CZI readers (czifile and pylibCZIrw) on random regions of real files.

One-off validation of the reader MerXen uses (``czifile``, BSD-3-Clause) against
Zeiss' own library (``pylibCZIrw``, LGPL-3.0, therefore never a MerXen
dependency). The script imports ``merxen`` nowhere and runs in an ephemeral
environment::

    uv run --no-project --python 3.12 --with "pylibCZIrw==<exact version>" \\
        --with "czifile==2026.10.14" --with imagecodecs --with numpy \\
        --with tifffile --with zarr \\
        python scripts/manual_registration/compare_czi_readers.py \\
        --manifest "$MERXEN_EVIDENCE_ROOT/manual_registration/czi_test_pairs.json" \\
        --out "$MERXEN_EVIDENCE_ROOT/manual_registration/czi_reader_comparison.json"

The manifest maps every alias to the reference and moving file (paths relative to
the test folder, ``$MERXEN_MANUAL_REGISTRATION_TEST_DIR`` or ``--data-dir``). For
every alias and role, every channel gets ``--rois`` random ROIs of 1024 x 1024
pixels, drawn uniformly inside the scene bounding box and re-drawn (up to 200
attempts) until the pylibCZIrw read has at least 50 % non-zero pixels. Both
readers read each ROI; the maximum absolute difference, the number of differing
pixels and the milliseconds per read are recorded. ``--converted-dir`` (holding
``<alias>_<role>.ome.tif``, as the prepare step writes them) adds the same ROI
read from level 0 of the converted OME-TIFF, compared with the pylibCZIrw maximum
over all Z planes (the converter max-projects a Z-stack); the two readers are
compared on plane Z=0. The output is keyed by
``<alias>_<role>`` only; it never holds file names.

Expectation: the pixels are equal (at most 1 DN apart for a JpegXr file). Any
difference is reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROI_SIZE = 1024
MIN_NONZERO_FRACTION = 0.5
MAX_ATTEMPTS = 200
ROLES = ("reference", "moving")


def _pylibczi_roi(doc: Any, x: int, y: int, c: int, z: int = 0) -> np.ndarray:
    """One ROI from pylibCZIrw (global coordinates, plane C=c, Z=z, T=0)."""
    data = doc.read(
        roi=(x, y, ROI_SIZE, ROI_SIZE),
        plane={"C": c, "Z": z, "T": 0},
        zoom=1.0,
        scene=0,
        background_pixel=(0, 0, 0),
    )
    return np.asarray(data[..., 0])


def _pylibczi_roi_max_over_z(
    doc: Any, x: int, y: int, c: int, z_range: tuple[int, int]
) -> np.ndarray:
    """One ROI from pylibCZIrw, maximum over Z as the converter writes it."""
    planes = [_pylibczi_roi(doc, x, y, c, z) for z in range(*z_range)]
    return np.maximum.reduce(planes)


def _czifile_roi(scene: Any, x: int, y: int, c: int) -> np.ndarray:
    """One ROI from czifile, with the call MerXen's converter uses."""
    dims = tuple(scene.dims)
    start = dict(zip(dims, scene.start, strict=True))
    selection: dict[str, int] = {}
    if "C" in dims:
        selection["C"] = int(start["C"]) + c
    if "Z" in dims:
        selection["Z"] = int(start["Z"])
    try:
        data = scene(roi=(x, y, ROI_SIZE, ROI_SIZE), **selection).asarray(
            fillvalue=0, maxworkers=1
        )
    except ValueError as exc:
        if "matches no subblocks" in str(exc):
            return np.zeros((ROI_SIZE, ROI_SIZE), dtype=scene.dtype)
        raise
    return np.asarray(data).reshape(ROI_SIZE, ROI_SIZE)


def _timed(function: Any, *args: Any) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter() - started) * 1000.0


def _compare_file(
    path: Path,
    *,
    n_rois: int,
    seed: int,
    converted: Path | None,
    thumbnail_dir: Path | None,
    key: str,
    thumbnail_channel: int,
) -> dict[str, Any]:
    import czifile
    from pylibCZIrw import czi as pyczi

    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "czifile_version": czifile.__version__,
        "channels": {},
    }
    tif = None
    converted_levels: Any = None
    if converted is not None and converted.is_file():
        import tifffile
        import zarr

        tif = tifffile.TiffFile(converted)
        converted_levels = zarr.open(tif.series[0].aszarr(level=0), mode="r")
    with (
        pyczi.open_czi(str(path)) as doc,
        czifile.CziFile(str(path), memmap=True, squeeze=False) as czi,
    ):
        rect = doc.scenes_bounding_rectangle[0]
        bbox_x, bbox_y, bbox_w, bbox_h = rect.x, rect.y, rect.w, rect.h
        scene_key = 0 if 0 in czi.scenes else next(iter(czi.scenes))
        scene = czi.scenes[scene_key]
        pixel_types = {int(k): str(v) for k, v in doc.pixel_types.items()}
        try:
            compression = str(czi.filtered_subblock_directory[0].compression)
        except Exception:  # noqa: BLE001 - informative field only
            compression = None
        result.update(
            {
                "pixel_types": pixel_types,
                "compression": compression,
                "bbox_size_xy": [int(bbox_w), int(bbox_h)],
            }
        )
        n_channels = len(pixel_types)
        z_start, z_end = doc.total_bounding_box.get("Z", (0, 1))
        result["z_planes"] = int(z_end - z_start)
        if thumbnail_dir is not None:
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            thumb = doc.read(
                plane={"C": thumbnail_channel, "Z": 0, "T": 0},
                zoom=1.0 / 48.0,
                scene=0,
            )
            np.save(thumbnail_dir / f"{key}_dapi_thumb.npy", np.asarray(thumb[..., 0]))
        for channel in range(n_channels):
            rows = []
            for _ in range(n_rois):
                for _attempt in range(MAX_ATTEMPTS):
                    x = int(rng.integers(bbox_x, bbox_x + max(1, bbox_w - ROI_SIZE)))
                    y = int(rng.integers(bbox_y, bbox_y + max(1, bbox_h - ROI_SIZE)))
                    reference, ms_lib = _timed(_pylibczi_roi, doc, x, y, channel)
                    if float(np.mean(reference != 0)) >= MIN_NONZERO_FRACTION:
                        break
                else:
                    logger.warning(
                        "%s channel %d: no ROI with >= 50 %% tissue in %d draws",
                        key,
                        channel,
                        MAX_ATTEMPTS,
                    )
                other, ms_czifile = _timed(_czifile_roi, scene, x, y, channel)
                diff = np.abs(reference.astype(np.int64) - other.astype(np.int64))
                row: dict[str, Any] = {
                    "max_abs_diff": int(diff.max()),
                    "n_diff_pixels": int((diff > 0).sum()),
                    "ms_czifile": ms_czifile,
                    "ms_pylibczirw": ms_lib,
                }
                if converted_levels is not None:
                    local_x, local_y = x - bbox_x, y - bbox_y
                    started = time.perf_counter()
                    rows_slice = slice(local_y, local_y + ROI_SIZE)
                    cols_slice = slice(local_x, local_x + ROI_SIZE)
                    if converted_levels.ndim == 2:  # a single channel: axes YX
                        window = np.asarray(converted_levels[rows_slice, cols_slice])
                    else:
                        window = np.asarray(
                            converted_levels[channel, rows_slice, cols_slice]
                        )
                    row["ms_ome_tiff"] = (time.perf_counter() - started) * 1000.0
                    padded = np.zeros((ROI_SIZE, ROI_SIZE), dtype=window.dtype)
                    padded[: window.shape[0], : window.shape[1]] = window
                    # The converter max-projects the Z planes, so the OME-TIFF is
                    # compared with the pylibCZIrw maximum over Z (plane Z=0 for a
                    # single-plane file, as read above).
                    projected = (
                        _pylibczi_roi_max_over_z(doc, x, y, channel, (z_start, z_end))
                        if z_end - z_start > 1
                        else reference
                    )
                    tiff_diff = np.abs(
                        projected.astype(np.int64) - padded.astype(np.int64)
                    )
                    row["tiff_max_abs_diff"] = int(tiff_diff.max())
                    row["tiff_n_diff_pixels"] = int((tiff_diff > 0).sum())
                rows.append(row)
            result["channels"][str(channel)] = {
                "n_rois": len(rows),
                "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                "mismatch_fraction": float(
                    sum(r["n_diff_pixels"] for r in rows) / (len(rows) * ROI_SIZE**2)
                ),
                "mean_ms_czifile": float(np.mean([r["ms_czifile"] for r in rows])),
                "mean_ms_pylibczirw": float(
                    np.mean([r["ms_pylibczirw"] for r in rows])
                ),
                "rois": rows,
            }
            if converted_levels is not None:
                result["channels"][str(channel)]["tiff_max_abs_diff"] = max(
                    r["tiff_max_abs_diff"] for r in rows
                )
                result["channels"][str(channel)]["mean_ms_ome_tiff"] = float(
                    np.mean([r["ms_ome_tiff"] for r in rows])
                )
    if tif is not None:
        tif.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare czifile and pylibCZIrw on random ROIs of CZI files."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder of the CZI files (default: $MERXEN_MANUAL_REGISTRATION_TEST_DIR).",
    )
    parser.add_argument("--rois", type=int, default=20, help="ROIs per channel.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=None,
        help="Folder of <alias>_<role>.ome.tif to compare against too.",
    )
    parser.add_argument(
        "--thumbnails",
        type=Path,
        default=None,
        help="Also write 1/48-zoom thumbnails (.npy) of one channel per file here.",
    )
    parser.add_argument("--thumbnail-channel", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit status (0; differences are reported in the output, not hidden by
        the exit code).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_dir = args.data_dir
    if data_dir is None:
        value = os.environ.get("MERXEN_MANUAL_REGISTRATION_TEST_DIR")
        if not value:
            raise SystemExit(
                "give --data-dir or set MERXEN_MANUAL_REGISTRATION_TEST_DIR"
            )
        data_dir = Path(value)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output: dict[str, Any] = {}
    for alias, entry in manifest["pairs"].items():
        for role in ROLES:
            key = f"{alias}_{role}"
            converted = (
                None
                if args.converted_dir is None
                else args.converted_dir / f"{key}.ome.tif"
            )
            logger.info("comparing %s", key)
            output[key] = _compare_file(
                data_dir / entry[role],
                n_rois=args.rois,
                seed=args.seed,
                converted=converted,
                thumbnail_dir=args.thumbnails,
                key=key,
                thumbnail_channel=args.thumbnail_channel,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    worst = max(
        (
            channel["max_abs_diff"]
            for entry in output.values()
            for channel in entry["channels"].values()
        ),
        default=0,
    )
    print(f"wrote {args.out}; largest difference between the readers: {worst} DN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
