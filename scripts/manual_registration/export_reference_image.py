#!/usr/bin/env python
"""Export one channel of a SpatialData image as a pyramidal OME-TIFF.

The manual registration reads images from OME-TIFF files. A Xenium
``morphology_focus`` image lives inside a SpatialData store, so a ground-truth
test that registers a post-Xenium IF image to it first exports the DAPI channel
to a pyramidal OME-TIFF with the pixel size recorded in the file.

Xenium stores keep ``morphology_focus`` in pixel units (an identity ``global``
transformation), so the pixel size cannot be read from the store: pass
``--pixel-size-um`` (0.2125 for Xenium morphology images). The export is skipped
when ``--out`` already holds the same export (a sidecar records what it was made
from).

Example::

    python scripts/manual_registration/export_reference_image.py \\
        --store "$MERXEN_RESULTS_ROOT/<pair>/xenium/latest/latest_spatialdata.zarr" \\
        --channel DAPI --pixel-size-um 0.2125 \\
        --out "$MERXEN_TASK_TMP_DIR/manual_registration/reference_dapi.ome.tif"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import spatialdata as sd
import xarray as xr
from spatialdata.transformations import get_transformation

from merxen import local_config
from merxen.io.ome_tiff_writer import OmeTiffWriteResult, write_pyramidal_ome_tiff
from merxen.manual_registration.schema import canonical_json, file_fingerprint

logger = logging.getLogger(__name__)

_SCALE_TOLERANCE = 0.01


def _level0(element: Any) -> Any:
    """Level 0 of an image element (a DataArray, or the ``scale0`` of a DataTree)."""
    if isinstance(element, xr.DataArray):
        return element
    dataset = element["scale0"].ds
    return dataset[next(iter(dataset.data_vars))]


def _channel_index(level0: Any, channel: str) -> int:
    names = [str(name) for name in level0.coords["c"].values]
    lowered = [name.lower() for name in names]
    if channel.lower() not in lowered:
        raise ValueError(f"channel {channel!r} not found; the image has {names}")
    return lowered.index(channel.lower())


def _global_scale(element: Any, image_key: str) -> tuple[float, float] | None:
    """The ``(x, y)`` scale of the element's ``global`` transformation, if a pure Scale.

    Raises:
        ValueError: If the transformation is not a pure scale (or has unequal
            axes scales).
    """
    transformation = get_transformation(element, "global")
    if transformation is None:
        return None
    matrix = np.asarray(
        transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    )
    off_diagonal = np.abs(matrix[:2, :2] - np.diag(np.diag(matrix[:2, :2]))).max()
    if off_diagonal > 1e-9 or np.abs(matrix[:2, 2]).max() > 1e-9:
        raise ValueError(
            f"{image_key} has a 'global' transformation that is not a pure scale; "
            "pass --pixel-size-um"
        )
    scale_x, scale_y = float(matrix[0, 0]), float(matrix[1, 1])
    if not math.isclose(scale_x, scale_y, rel_tol=1e-6):
        raise ValueError(
            f"{image_key} has unequal x and y scales ({scale_x}, {scale_y}); "
            "pass --pixel-size-um"
        )
    return scale_x, scale_y


def _pixel_size(
    element: Any, image_key: str, pixel_size_um: float | None
) -> tuple[float, float | None]:
    """Resolve the pixel size to write and the store's own scale (for the log).

    A given ``pixel_size_um`` wins: a ``global`` transformation that is not a pure
    scale is then only logged, never an error.
    """
    if pixel_size_um is not None:
        if pixel_size_um <= 0 or not math.isfinite(pixel_size_um):
            raise ValueError(f"pixel_size_um must be > 0, got {pixel_size_um}")
        try:
            store_scale = _global_scale(element, image_key)
        except ValueError:
            logger.info(
                "the 'global' transformation of %s is not an isotropic pure scale; "
                "using the given %.6g um/px",
                image_key,
                pixel_size_um,
            )
            store_scale = None
        if (
            store_scale is not None
            and store_scale[0] != 1.0
            and abs(store_scale[0] / pixel_size_um - 1.0) > _SCALE_TOLERANCE
        ):
            logger.warning(
                "the store's 'global' scale of %s is %.6g um/px but %.6g was "
                "given; using the given value",
                image_key,
                store_scale[0],
                pixel_size_um,
            )
        return float(pixel_size_um), None if store_scale is None else store_scale[0]
    store_scale = _global_scale(element, image_key)
    if store_scale is None or store_scale == (1.0, 1.0):
        raise ValueError(
            f"{image_key} has an identity/pixel-unit 'global' transformation, so "
            "its pixel size is unknown; pass --pixel-size-um (pixel_size_um=...) "
            "with the true value, e.g. 0.2125 for Xenium morphology images"
        )
    return store_scale[0], store_scale[0]


def export_channel(
    store: Path,
    *,
    image_key: str,
    channel: str,
    out: Path,
    pixel_size_um: float | None,
    image_name: str | None = None,
    workers: int = 8,
    tmp_dir: Path | None = None,
) -> OmeTiffWriteResult | None:
    """Write one channel of a store image as a pyramidal OME-TIFF.

    Args:
        store: SpatialData store.
        image_key: Image element (a ``(c, y, x)`` image).
        channel: Channel name (case-insensitive).
        out: Output ``.ome.tif``.
        pixel_size_um: Pixel size to record; ``None`` reads the store's ``global``
            scale, which must then be a pure scale other than the identity.
        image_name: OME image name (default ``<image_key>_<channel>``).
        workers: Compression threads.
        tmp_dir: Temporary directory of the writer.

    Returns:
        The write result, or ``None`` when ``out`` is already an up-to-date export.

    Raises:
        ValueError: If the channel is missing or the pixel size cannot be resolved
            (no file is written then). A ``global`` transformation that is not a
            pure scale is an error only when ``pixel_size_um`` is ``None``.
    """
    sdata = sd.read_zarr(Path(store), selection=("images",))
    element = sdata.images[image_key]
    level0 = _level0(element)
    index = _channel_index(level0, channel)
    pixel_size, store_scale = _pixel_size(element, image_key, pixel_size_um)
    # The store's own spelling of the channel, not the caller's (case-insensitive) one.
    store_channel = level0.coords["c"].values[index].item()
    out = Path(out)
    sidecar = out.with_suffix(".export.json")
    height, width = int(level0.sizes["y"]), int(level0.sizes["x"])
    dtype = np.dtype(level0.dtype)
    key = {
        "shape_yx": [height, width],
        "dtype": str(dtype),
        "pixel_size_um": pixel_size,
        "image_key": image_key,
        "channel": store_channel,
    }
    if out.is_file() and sidecar.is_file():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if recorded.get("key") == key and recorded.get("fingerprint") == (
            file_fingerprint(out)
        ):
            logger.info("%s is up to date", out)
            return None
    logger.info(
        "exporting %s/%s (%d x %d px, %.6g um/px; store scale %s) to %s",
        image_key,
        store_channel,
        height,
        width,
        pixel_size,
        store_scale,
        out,
    )
    data = level0.isel(c=index).data

    def read_band(_channel: int, y0: int, band: int) -> np.ndarray:
        return np.asarray(data[y0 : y0 + band].compute())

    out.parent.mkdir(parents=True, exist_ok=True)
    result = write_pyramidal_ome_tiff(
        read_band,
        shape_cyx=(1, height, width),
        dtype=dtype,
        channel_names=[str(store_channel)],
        pixel_size_um=pixel_size,
        image_name=image_name or f"{image_key}_{store_channel}",
        output_path=out,
        workers=workers,
        tmp_dir=tmp_dir,
    )
    sidecar.write_text(
        canonical_json({"key": key, "fingerprint": file_fingerprint(out)}) + "\n",
        encoding="utf-8",
    )
    return result


def _refuse_inside_repository(out: Path) -> None:
    root = local_config.checkout_root()
    if root is not None and out.resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"--out must not be inside the repository checkout ({root})")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Export one SpatialData image channel as a pyramidal OME-TIFF."
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--image-key", default="morphology_focus")
    parser.add_argument("--channel", default="DAPI")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output .ome.tif (outside the repo)."
    )
    parser.add_argument("--image-name", default=None)
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Pixel size to record (required when the store's global transformation "
        "is the identity, as for Xenium morphology_focus: 0.2125).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=local_config.path("MERXEN_TASK_TMP_DIR"),
        help="Writer temporary directory (default: $MERXEN_TASK_TMP_DIR).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _refuse_inside_repository(args.out)
    result = export_channel(
        args.store,
        image_key=args.image_key,
        channel=args.channel,
        out=args.out,
        pixel_size_um=args.pixel_size_um,
        image_name=args.image_name,
        workers=args.workers,
        tmp_dir=args.tmp_dir,
    )
    print("up to date" if result is None else f"wrote {result.bytes_written} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
