#!/usr/bin/env python
"""Compare physical density kernels with DAPI/18S at existing pilot locations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
from matplotlib.axes import Axes

from merxen.config import AteraTranscriptDensityConfig
from merxen.io.image_source import image_to_cyx
from merxen.platforms import platform_spec
from merxen.segmentation.transcript_density import (
    accumulate_density_counts,
    native_density_geometry,
    reconstruct_density_window,
    smooth_density_counts,
)


def _display(plane: np.ndarray) -> np.ndarray:
    low, high = np.percentile(plane, (2, 99))
    return np.asarray(
        np.clip((plane.astype(float) - low) / max(high - low, 1e-8), 0, 1)
    )


def _image(axis: Axes, data: np.ndarray, title: str, size_um: float) -> None:
    axis.imshow(data, cmap="gray", extent=(0, size_um, size_um, 0))
    axis.set_title(title, fontsize=12)
    axis.axis("off")
    axis.plot(
        [size_um - 25, size_um - 5], [size_um - 8] * 2, color="white", linewidth=3
    )
    axis.text(
        size_um - 15, size_um - 12, "20 µm", color="white", ha="center", fontsize=10
    )


def main() -> int:
    """Write matched overlay, separate-channel and kernel-comparison PNGs.

    Returns:
        Zero after the plots and their exact source/bounds record are written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigma-um", type=float, default=0.15)
    parser.add_argument("--bin-size-px", type=int, default=1)
    parser.add_argument(
        "--compare-sigmas", type=float, nargs=3, default=[0.15, 0.25, 0.5]
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    geometry = native_density_geometry(args.source_zarr)
    image = sd.read_zarr(args.source_zarr, selection=("images",)).images[
        platform_spec("ATERA").native_image_key
    ]
    native = image_to_cyx(image)
    pilot = json.loads(args.pilot_manifest.read_text())
    rows = [pilot["crops"][index] for index in (0, 3, 4)]
    sigmas = tuple(args.compare_sigmas)
    overlay, axes = plt.subplots(1, 3, figsize=(14, 5))
    channels, separate = plt.subplots(3, 3, figsize=(12, 12))
    records = []
    for index, row in enumerate(rows):
        kernels, compare = plt.subplots(2, 3, figsize=(12, 8))
        y0, y1, x0, x1 = row["bbox_y0_y1_x0_x1"]
        padding = math.ceil(4 * max(sigmas) / geometry.pixel_size_um) + 8
        window = (
            max(0, y0 - padding),
            min(geometry.height, y1 + padding),
            max(0, x0 - padding),
            min(geometry.width, x1 + padding),
        )
        config = AteraTranscriptDensityConfig(
            source_zarr_path=args.source_zarr,
            output_dir=args.output_dir,
            sigma_um=args.sigma_um,
            bin_size_px=args.bin_size_px,
        )
        counts = accumulate_density_counts(
            args.source_zarr, geometry, config, window=window
        )
        raw = native.sel(c=["DAPI", "18S"]).data[:, y0:y1, x0:x1].compute()
        dapi, rna = map(_display, raw)
        size_um = (y1 - y0) * geometry.pixel_size_um
        for sigma in sigmas:
            smoothed = smooth_density_counts(
                counts, geometry, config.model_copy(update={"sigma_um": sigma})
            )
            density = reconstruct_density_window(
                smoothed,
                origin=(counts.y0, counts.x0),
                bin_size_px=config.bin_size_px,
                window=(y0, y1, x0, x1),
            )
            column = sigmas.index(sigma)
            _image(
                compare[0, column],
                _display(density),
                f"Density · σ {sigma:g} µm",
                size_um,
            )
            _image(
                compare[1, column],
                np.stack([_display(density), rna, dapi], -1),
                f"Overlay · σ {sigma:g} µm",
                size_um,
            )
            if sigma == args.sigma_um:
                view = _display(density)
                _image(
                    axes[index],
                    np.stack([view, rna, dapi], -1),
                    f"ROI {index + 1}",
                    size_um,
                )
                for column, (data, name) in enumerate(
                    zip(
                        [dapi, rna, view],
                        ["DAPI", "18S RNA", "Transcript density"],
                        strict=True,
                    )
                ):
                    _image(
                        separate[index, column],
                        data,
                        f"ROI {index + 1} · {name}",
                        size_um,
                    )
                np.savez_compressed(
                    args.output_dir / f"roi_{index + 1}.npz",
                    dapi=raw[0],
                    rna18s=raw[1],
                    density=density,
                )
                records.append(
                    {
                        "roi": index + 1,
                        "bbox_y0_y1_x0_x1": [y0, y1, x0, x1],
                        "n_transcripts_with_halo": counts.n_binned,
                        "density_percentiles": np.percentile(
                            density, [0, 2, 50, 98, 99, 100]
                        ).tolist(),
                    }
                )
        kernels.suptitle(
            f"ROI {index + 1} · native-pixel density smoothing comparison", fontsize=15
        )
        kernels.text(
            0.5,
            0.015,
            "Native pixel bins; Q20 genes, all z; per-panel p2–p99 display scaling.",
            ha="center",
            fontsize=10,
        )
        kernels.tight_layout(rect=(0, 0.04, 1, 0.95))
        kernels.savefig(args.output_dir / f"smoothing-roi-{index + 1}.png", dpi=160)
        plt.close(kernels)
    overlay.suptitle(
        "Atera · DAPI (blue), 18S RNA (green), transcript density (red)", fontsize=15
    )
    channels.suptitle(
        "Atera · individual channels at the same native locations", fontsize=15
    )
    for figure, name in [
        (overlay, "overlays.png"),
        (channels, "individual-channels.png"),
    ]:
        figure.text(
            0.5,
            0.015,
            "All Q20 gene transcripts, assigned + unassigned; z summed. "
            "Per-ROI/channel p2–p99 display scaling.",
            ha="center",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.95))
        figure.savefig(args.output_dir / name, dpi=160)
        plt.close(figure)
    (args.output_dir / "sources.json").write_text(
        json.dumps(
            {
                "source_zarr": str(args.source_zarr.resolve()),
                "pixel_size_um": geometry.pixel_size_um,
                "bin_size_px": args.bin_size_px,
                "sigma_um": args.sigma_um,
                "min_qv": 20,
                "regions": records,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
